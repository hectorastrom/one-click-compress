"""
LLM Agent — the brain of the agentic quantization pipeline.

Handles:
- LLM client (OpenAI / Anthropic) with structured JSON output
- Planning prompt: given a model scan report, propose quantization configs
- Analysis prompt: given evaluation results, recommend best configs or propose new ones
- Orchestrator: the main scan -> plan -> quantize -> eval -> iterate loop
"""

from __future__ import annotations

import json
import logging
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .scanner import ModelReport, scan_model
from .quantizers.base import (
    QuantizationConfig,
    QuantizationResult,
    QuantizerRegistry,
    build_default_registry,
)
from .evaluator import EvalResult, evaluate, measure_model_size_mb
from .pareto import find_pareto_optimal, plot_pareto
from .trace import AgentTrace

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic."""

    def __init__(self, provider: str = "openai", api_key: str = "", model: str = ""):
        self.provider = provider.lower()
        self.api_key = api_key

        if self.provider == "openai":
            self.model = model or "gpt-4o"
        elif self.provider == "anthropic":
            self.model = model or "claude-sonnet-4-20250514"
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat message and return the response text."""
        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return response.choices[0].message.content

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        from anthropic import Anthropic

        client = Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

PLAN_SYSTEM_PROMPT = """You are an expert in neural network model quantization and compression.
You analyze model architectures and propose optimal quantization configurations.

You MUST respond with valid JSON in this exact format:
{
  "reasoning": "Your detailed analysis: (1) which layers are most/least sensitive based on the sensitivity_table, (2) what the Beta distribution shapes tell you about each layer's weight distribution, (3) why you chose each config...",
  "configs": [
    {
      "method": "<method_name>",
      "target_layers": "all",
      "extra_params": {},
      "per_layer_config": {}
    }
  ]
}

Available quantization methods:
- "dynamic_int8": PyTorch dynamic quantization (INT8 weights, no calibration needed). Works on Linear and Conv layers. Good baseline.
- "static_int8": PyTorch static quantization (INT8 weights + activations, needs calibration). Best for CNNs. Requires architecture_hint=="cnn" or "mixed".
- "fp16": Float16 half-precision. Simple, always applicable, ~2x compression.
- "mixed_precision": CRITICAL METHOD — applies DIFFERENT precision per layer based on sensitivity analysis. You MUST propose at least one mixed_precision config. Use "per_layer_config" to assign per-layer precision, and "extra_params.default_precision" for unlisted layers.
  Valid precision values: "fp32" (keep original), "fp16" (half precision), "int8" (dynamic quantized).
  Example:
  {
    "method": "mixed_precision",
    "target_layers": "all",
    "extra_params": {"default_precision": "int8"},
    "per_layer_config": {
      "vision_model.encoder.layers.0.self_attn.q_proj": {"precision": "fp32"},
      "vision_model.encoder.layers.0.self_attn.k_proj": {"precision": "fp16"},
      "visual_projection": {"precision": "fp32"}
    }
  }

## Understanding the sensitivity_table

The scan report includes a "sensitivity_table" — a per-layer analysis sorted by sensitivity (most sensitive first). Each entry has:
- **name**: layer name (use these EXACT names in per_layer_config)
- **type**: layer type (Linear, Conv2d, etc.)
- **params**: parameter count
- **sensitivity**: 0-1 score. THIS IS THE KEY METRIC FOR YOUR DECISIONS:
  - 0.0–0.2: very robust → safe for int8 or even skip quantization overhead
  - 0.2–0.5: moderate → int8 is safe
  - 0.5–0.8: sensitive → keep at fp16 or fp32
  - 0.8–1.0: very sensitive → MUST keep at fp32
- **beta_alpha, beta_beta**: Beta distribution shape of normalized weights
  - Both > 2: bell-shaped, well-behaved → quantize aggressively
  - Either < 1: skewed/U-shaped, outlier-prone → preserve precision
- **kurtosis**: > 3 means heavy tails / outlier weights → more precision needed

The report also includes "sensitivity_summary" with counts of high/low sensitivity layers.

## Rules

- Only propose methods in the "applicable_methods" list.
- You MUST propose 3-5 configs, and AT LEAST ONE must be "mixed_precision" with a per_layer_config that references actual layer names from the sensitivity_table.
- For mixed_precision: put all layers with sensitivity > 0.5 at "fp32", layers with sensitivity 0.2-0.5 at "fp16", and the rest at "int8" (via default_precision).
- Propose configs from conservative (fp16 only) to aggressive (all int8) to smart (mixed_precision using sensitivity).
- In your reasoning, explicitly cite which layers are most sensitive, their Beta distribution parameters, and what precision you assigned them.
"""

PLAN_USER_TEMPLATE = """Here is the scan report for model "{model_name}":

{scan_report_json}

IMPORTANT: Look at the "sensitivity_table" in the report. It contains per-layer Beta distribution analysis.
Use the exact layer names from the sensitivity_table when building your mixed_precision per_layer_config.

You MUST:
1. Analyze the sensitivity_table — identify the most sensitive layers (highest sensitivity score) and explain WHY they are sensitive (Beta params, kurtosis).
2. Propose a mixed_precision config that assigns fp32 to sensitive layers and int8 to robust layers.
3. Also propose uniform configs (fp16, dynamic_int8) as comparison baselines.
"""

ANALYZE_SYSTEM_PROMPT = """You are an expert in neural network quantization analyzing experimental results.

You MUST respond with valid JSON in this exact format:
{
  "analysis": "Your analysis of the results...",
  "best_config": "name of the best config considering all trade-offs",
  "pareto_optimal": ["list", "of", "pareto", "optimal", "config", "names"],
  "recommendation": "Final recommendation for the user...",
  "try_more": false,
  "new_configs": []
}

If you want to try additional configurations, set "try_more": true and provide "new_configs" in the same format as the planning phase.
The "new_configs" format supports mixed_precision with per_layer_config:
{
  "method": "mixed_precision",
  "target_layers": "all",
  "extra_params": {"default_precision": "int8"},
  "per_layer_config": {"layer_name": {"precision": "fp32"}}
}
Only suggest new configs if you believe there's a gap in the explored trade-off space.
Consider whether a mixed_precision config could improve the Pareto frontier by
keeping sensitive layers at higher precision while compressing others aggressively.
"""

ANALYZE_USER_TEMPLATE = """Here are the quantization experiment results for model "{model_name}":

Original model size: {original_size_mb:.2f} MB

Results:
{results_json}

Analyze these results. Identify which configurations are Pareto-optimal (best accuracy for a given size, or smallest size for a given accuracy).
Recommend the best overall configuration and explain why.
If there are gaps in the trade-off space worth exploring, suggest additional configs — particularly mixed_precision configs
that could leverage per-layer sensitivity information to achieve better accuracy/size trade-offs.
"""

# --- Critic agent: reviews the strategist's proposals ---
CRITIC_SYSTEM_PROMPT = """You are a critical reviewer of neural network quantization strategies.
Another agent has proposed quantization configurations. Your job is to:
1. Identify weaknesses or blind spots in the proposals
2. Check if the Beta distribution analysis was properly used
3. Suggest improvements to the mixed_precision config (more/fewer protected layers?)
4. Flag any configs that are likely to fail or produce poor results

You MUST respond with valid JSON:
{
  "critique": "Your detailed critique of the proposals...",
  "issues": ["list of specific issues found"],
  "improved_mixed_precision": {
    "method": "mixed_precision",
    "target_layers": "all",
    "extra_params": {"default_precision": "int8"},
    "per_layer_config": {"layer_name": {"precision": "fp32"}}
  },
  "drop_configs": ["list of config methods to drop, if any"],
  "add_configs": []
}

Focus especially on the mixed_precision config:
- Are the right layers being protected? Check the sensitivity scores.
- Are there layers with sensitivity > 0.3 that should be protected but aren't?
- Are there layers being protected unnecessarily (sensitivity < 0.1)?
- Would a different split between fp32 and int8 layers produce better results?
"""

CRITIC_USER_TEMPLATE = """The Strategist proposed these quantization configs for model "{model_name}":

{proposed_configs_json}

Here is the model's sensitivity analysis for reference:
{sensitivity_summary_json}

Top 15 most sensitive layers:
{top_sensitive_json}

Review these proposals critically. Is the mixed_precision config optimal?
Should different layers be protected? Are there missing strategies?
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class QuantizationOrchestrator:
    """
    Multi-agent orchestrator: Scanner -> Strategist -> Critic -> Executor -> Analyst.
    Records every step in an AgentTrace for replay.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        registry: QuantizerRegistry | None = None,
        max_iterations: int = 2,
        output_dir: str = "results",
        show_examples: bool = False,
    ):
        self.llm = llm_client
        self.registry = registry or build_default_registry()
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.show_examples = show_examples
        self.all_results: list[EvalResult] = []
        self.quantized_models: dict[str, Any] = {}  # config_name -> nn.Module
        self.trace: AgentTrace | None = None

    def run(
        self,
        model: Any,  # nn.Module
        model_name: str,
        test_loader: Any,  # DataLoader
        calibration_loader: Any | None = None,
        device: Any = None,  # torch.device
        input_shape: tuple[int, ...] = (1, 3, 32, 32),
        max_eval_batches: int | None = None,
    ) -> list[EvalResult]:
        """
        Execute the full multi-agent quantization pipeline with trace logging.

        Agent flow: Scanner -> Strategist -> Critic -> Executor -> Analyst
        Every step is recorded in self.trace for replay.

        Returns a list of EvalResults for all tried configurations.
        """
        import torch
        import time as _time
        import os

        pipeline_start = _time.perf_counter()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        model.eval()

        # Initialize trace
        self.trace = AgentTrace(model_name=model_name)

        # ================================================================
        # STEP 1: Scanner Agent — analyze model architecture
        # ================================================================
        console.print(Panel("[bold]Agent 1: Scanner — Analyzing model architecture...[/bold]"))
        t0 = _time.perf_counter()

        report = scan_model(model, model_name)
        original_size_mb = measure_model_size_mb(model)
        report_dict = report.to_dict()

        scan_duration = (_time.perf_counter() - t0) * 1000
        console.print(report.to_summary_string())

        # Build scan summary for trace
        sens_summary = report_dict.get("sensitivity_summary", {})
        top_sensitive = report_dict.get("sensitivity_table", [])[:10]
        scan_output = (
            f"Model: {model_name}\n"
            f"Parameters: {report.total_params:,}\n"
            f"Size: {report.total_size_mb:.1f} MB\n"
            f"Architecture: {report.architecture_hint}\n"
            f"Applicable methods: {report.applicable_methods}\n\n"
            f"Beta Distribution Analysis:\n"
            f"  Analyzed layers: {sens_summary.get('total_analyzed_layers', 0)}\n"
            f"  Avg sensitivity: {sens_summary.get('avg_sensitivity', 0):.3f}\n"
            f"  Max sensitivity: {sens_summary.get('max_sensitivity', 0):.3f}\n"
            f"  High-sensitivity layers: {sens_summary.get('high_sensitivity_layers', [])}\n\n"
            f"Top 10 most sensitive layers:\n"
        )
        for entry in top_sensitive:
            scan_output += (
                f"  {entry['name']}: sensitivity={entry['sensitivity']:.3f}, "
                f"Beta(a={entry['beta_alpha']:.2f}, b={entry['beta_beta']:.2f}), "
                f"kurtosis={entry['kurtosis']:.2f}\n"
            )

        self.trace.add_step(
            agent_role="Scanner",
            agent_icon="🔍",
            title="Model Architecture & Beta Distribution Analysis",
            input_summary=f"Model: {model_name} ({report.total_params:,} params)",
            output_text=scan_output,
            duration_ms=scan_duration,
            data={"sensitivity_summary": sens_summary},
        )

        # ================================================================
        # STEP 2: Baseline evaluation
        # ================================================================
        console.print(Panel("[bold]Baseline Evaluation...[/bold]"))
        t0 = _time.perf_counter()
        baseline = evaluate(
            model=model, test_loader=test_loader, device=device,
            original_size_mb=original_size_mb, config_name="baseline_fp32",
            input_shape=input_shape, max_eval_batches=max_eval_batches,
        )
        baseline_duration = (_time.perf_counter() - t0) * 1000
        self.all_results.append(baseline)
        self._print_result(baseline)
        self.quantized_models["baseline_fp32"] = model

        self.trace.add_step(
            agent_role="Executor",
            agent_icon="⚡",
            title="Baseline FP32 Evaluation",
            input_summary="Running original model on test set",
            output_text=(
                f"Baseline results:\n"
                f"  Accuracy: {baseline.accuracy_pct:.2f}%\n"
                f"  Size: {baseline.size_mb:.1f} MB\n"
                f"  Latency: {baseline.latency_ms:.1f} ms\n"
            ),
            duration_ms=baseline_duration,
            data={"baseline": baseline.to_dict()},
        )

        # ================================================================
        # STEP 3: Strategist Agent — propose quantization configs
        # ================================================================
        console.print(Panel("[bold]Agent 2: Strategist — Proposing quantization strategies...[/bold]"))
        t0 = _time.perf_counter()
        configs, strategist_reasoning = self._plan_with_trace(report)
        strategist_duration = (_time.perf_counter() - t0) * 1000

        config_descriptions = []
        for c in configs:
            desc = f"  - {c.method}"
            if c.per_layer_config:
                desc += f" (per_layer_config: {len(c.per_layer_config)} layers specified)"
            config_descriptions.append(desc)

        self.trace.add_step(
            agent_role="Strategist",
            agent_icon="🧠",
            title="Quantization Strategy Proposals",
            input_summary=f"Scan report with {sens_summary.get('total_analyzed_layers', 0)} layers analyzed",
            output_text=(
                f"{strategist_reasoning}\n\n"
                f"Proposed {len(configs)} configurations:\n"
                + "\n".join(config_descriptions)
            ),
            duration_ms=strategist_duration,
            data={"configs": [c.to_dict() for c in configs]},
        )

        # ================================================================
        # STEP 4: Critic Agent — review and improve proposals
        # ================================================================
        console.print(Panel("[bold]Agent 3: Critic — Reviewing proposals...[/bold]"))
        t0 = _time.perf_counter()
        configs, critic_output = self._critique_and_improve(configs, report)
        critic_duration = (_time.perf_counter() - t0) * 1000

        self.trace.add_step(
            agent_role="Critic",
            agent_icon="🔎",
            title="Critical Review & Improvement of Proposals",
            input_summary=f"Reviewing {len(configs)} proposed configs against sensitivity data",
            output_text=critic_output,
            duration_ms=critic_duration,
            data={"final_configs": [c.to_dict() for c in configs]},
        )

        # ================================================================
        # STEP 5: Executor — quantize and evaluate each config
        # ================================================================
        console.print(Panel("[bold]Agent 4: Executor — Running quantization experiments...[/bold]"))
        t0 = _time.perf_counter()
        exec_lines = []

        for i, config in enumerate(configs):
            console.print(
                f"\n  [{i+1}/{len(configs)}] Quantizing with: "
                f"[bold]{config.method}[/bold]"
            )

            backend = self.registry.get(config.method)
            if backend is None:
                console.print(f"  [red]No backend for '{config.method}', skipping.[/red]")
                exec_lines.append(f"[{config.method}] SKIPPED — no backend registered")
                continue

            import torch as _torch
            quant_result = backend.quantize(
                model=model, config=config, calibration_data=calibration_loader,
            )

            if not quant_result.success:
                console.print(f"  [red]Quantization failed: {quant_result.error}[/red]")
                self.all_results.append(EvalResult(
                    config_name=config.method,
                    accuracy_pct=0.0, size_mb=0.0,
                    latency_ms=float("inf"), compression_ratio=0.0,
                    error=quant_result.error,
                ))
                exec_lines.append(f"[{config.method}] FAILED: {quant_result.error}")
                continue

            eval_result = evaluate(
                model=quant_result.model, test_loader=test_loader, device=device,
                original_size_mb=original_size_mb, config_name=config.method,
                input_shape=input_shape, max_eval_batches=max_eval_batches,
                metadata=quant_result.metadata,
            )
            self.all_results.append(eval_result)
            self._print_result(eval_result)

            exec_lines.append(
                f"[{config.method}] accuracy={eval_result.accuracy_pct:.2f}%, "
                f"size={eval_result.size_mb:.1f} MB, "
                f"latency={eval_result.latency_ms:.1f} ms, "
                f"compression={eval_result.compression_ratio:.2f}x"
            )

            if self.show_examples:
                self.quantized_models[config.method] = quant_result.model
            else:
                del quant_result
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()

        exec_duration = (_time.perf_counter() - t0) * 1000

        # Build results table for trace
        results_table = f"{'Config':<20} {'Accuracy':>10} {'Size (MB)':>10} {'Latency':>10} {'Compress':>10}\n"
        results_table += "-" * 62 + "\n"
        for r in self.all_results:
            if r.error is None:
                results_table += (
                    f"{r.config_name:<20} {r.accuracy_pct:>9.2f}% {r.size_mb:>9.1f} "
                    f"{r.latency_ms:>9.1f} {r.compression_ratio:>9.2f}x\n"
                )

        self.trace.add_step(
            agent_role="Executor",
            agent_icon="⚡",
            title=f"Quantization Experiments — {len(configs)} configs tested",
            input_summary=f"Running {len(configs)} quantization configs",
            output_text="\n".join(exec_lines),
            duration_ms=exec_duration,
            data={"results_table": results_table},
        )

        # ================================================================
        # STEP 6: Analyst Agent — final analysis and recommendation
        # ================================================================
        console.print(Panel("[bold]Agent 5: Analyst — Final Analysis...[/bold]"))
        t0 = _time.perf_counter()
        recommendation = self._final_analysis_with_trace(report, original_size_mb)
        analyst_duration = (_time.perf_counter() - t0) * 1000

        self.trace.add_step(
            agent_role="Analyst",
            agent_icon="📊",
            title="Final Analysis & Recommendation",
            input_summary=f"Analyzing {len(self.all_results)} experiment results",
            output_text=recommendation,
            duration_ms=analyst_duration,
            data={"results_table": results_table},
        )

        self.trace.final_recommendation = recommendation
        self.trace.total_duration_ms = (_time.perf_counter() - pipeline_start) * 1000

        # ---- Save trace ----
        os.makedirs(self.output_dir, exist_ok=True)
        trace_path = os.path.join(self.output_dir, "agent_trace.json")
        self.trace.save(trace_path)
        console.print(f"\n[green]Agent trace saved to {trace_path}[/green]")

        # ---- Pareto plot ----
        try:
            valid_results = [r for r in self.all_results if r.error is None]
            if len(valid_results) >= 2:
                plot_path = os.path.join(self.output_dir, "pareto.png")
                plot_pareto(valid_results, save_path=plot_path)
                console.print(f"[green]Pareto plot saved to {plot_path}[/green]")
        except Exception as e:
            logger.warning(f"Could not generate Pareto plot: {e}")

        # ---- Prediction examples ----
        if self.show_examples and self.quantized_models:
            console.print(Panel("[bold]Generating prediction examples...[/bold]"))
            try:
                from .visualize import generate_prediction_examples, generate_failure_analysis
                os.makedirs(self.output_dir, exist_ok=True)

                examples_path = os.path.join(self.output_dir, "examples.png")
                generate_prediction_examples(
                    models=self.quantized_models, test_loader=test_loader,
                    device=device, n_samples=12, save_path=examples_path,
                )
                console.print(f"[green]Prediction examples saved to {examples_path}[/green]")

                failure_path = os.path.join(self.output_dir, "failure_analysis.png")
                generate_failure_analysis(
                    models=self.quantized_models, test_loader=test_loader,
                    device=device, n_correct=6, n_incorrect=6, save_path=failure_path,
                )
                console.print(f"[green]Failure analysis saved to {self.output_dir}/failure_analysis_*.png[/green]")
            except Exception as e:
                logger.warning(f"Could not generate prediction examples: {e}")

        return self.all_results

    def _plan_with_trace(self, report: ModelReport) -> tuple[list[QuantizationConfig], str]:
        """Strategist agent: propose quantization configs. Returns (configs, reasoning)."""
        console.print("  Strategist is analyzing sensitivity data...")

        report_dict = report.to_dict()
        user_prompt = PLAN_USER_TEMPLATE.format(
            model_name=report.model_name,
            scan_report_json=json.dumps(report_dict, indent=2, default=str),
        )

        try:
            response_text = self.llm.chat(PLAN_SYSTEM_PROMPT, user_prompt)
            response = self._parse_json_response(response_text)

            reasoning = response.get("reasoning", "No reasoning provided.")
            console.print(f"\n  [dim]Strategist: {reasoning[:300]}...[/dim]\n")

            raw_configs = response.get("configs", [])
            configs = [QuantizationConfig.from_dict(c) for c in raw_configs]

            available = set(self.registry.available_methods())
            applicable = set(report.applicable_methods)
            valid_methods = available & applicable

            filtered = [c for c in configs if c.method in valid_methods]
            if not filtered:
                filtered = self._default_configs(report)
                reasoning += "\n\n[Fallback: LLM proposed no valid configs, using defaults.]"

            console.print(f"  Proposed configs: {[c.method for c in filtered]}")
            return filtered, reasoning

        except Exception as e:
            logger.error(f"Strategist failed: {e}")
            return self._default_configs(report), f"Strategist failed: {e}. Using defaults."

    def _critique_and_improve(
        self, configs: list[QuantizationConfig], report: ModelReport
    ) -> tuple[list[QuantizationConfig], str]:
        """Critic agent: review proposals and improve the mixed_precision config."""
        console.print("  Critic is reviewing proposals...")

        report_dict = report.to_dict()
        sens_summary = report_dict.get("sensitivity_summary", {})
        sens_table = report_dict.get("sensitivity_table", [])

        user_prompt = CRITIC_USER_TEMPLATE.format(
            model_name=report.model_name,
            proposed_configs_json=json.dumps([c.to_dict() for c in configs], indent=2),
            sensitivity_summary_json=json.dumps(sens_summary, indent=2),
            top_sensitive_json=json.dumps(sens_table[:15], indent=2, default=str),
        )

        try:
            response_text = self.llm.chat(CRITIC_SYSTEM_PROMPT, user_prompt)
            response = self._parse_json_response(response_text)

            critique = response.get("critique", "No critique provided.")
            issues = response.get("issues", [])
            console.print(f"\n  [dim]Critic: {critique[:300]}...[/dim]")
            if issues:
                console.print(f"  [yellow]Issues found: {len(issues)}[/yellow]")
                for issue in issues[:3]:
                    console.print(f"    - {issue[:100]}")

            # Apply critic's improved mixed_precision config
            improved_mp = response.get("improved_mixed_precision")
            if improved_mp and isinstance(improved_mp, dict):
                improved_config = QuantizationConfig.from_dict(improved_mp)
                # Replace any existing mixed_precision config
                configs = [c for c in configs if c.method != "mixed_precision"]
                configs.append(improved_config)
                console.print(f"  [green]Critic improved the mixed_precision config "
                            f"({len(improved_mp.get('per_layer_config', {}))} per-layer rules)[/green]")

            # Drop configs the critic flagged
            drop = response.get("drop_configs", [])
            if drop:
                configs = [c for c in configs if c.method not in drop]
                console.print(f"  [yellow]Dropped configs: {drop}[/yellow]")

            # Add any new configs
            add_raw = response.get("add_configs", [])
            for raw in add_raw:
                configs.append(QuantizationConfig.from_dict(raw))

            output_text = f"Critique:\n{critique}\n\n"
            if issues:
                output_text += "Issues found:\n" + "\n".join(f"  - {i}" for i in issues) + "\n\n"
            output_text += f"Final config set: {[c.method for c in configs]}"

            console.print(f"\n  Final configs after review: {[c.method for c in configs]}")
            return configs, output_text

        except Exception as e:
            logger.error(f"Critic failed: {e}")
            return configs, f"Critic failed: {e}. Proceeding with original proposals."

    def _final_analysis_with_trace(self, report: ModelReport, original_size_mb: float) -> str:
        """Analyst agent: final analysis and recommendation. Returns recommendation text."""
        valid = [r for r in self.all_results if r.error is None]

        if not valid:
            return "No successful quantization results to analyze."

        # Print results table
        table = Table(title="Quantization Results")
        table.add_column("Config", style="bold")
        table.add_column("Accuracy (%)", justify="right")
        table.add_column("Size (MB)", justify="right")
        table.add_column("Latency (ms)", justify="right")
        table.add_column("Compression", justify="right")

        pareto_points = find_pareto_optimal(valid)
        for r in valid:
            style = "green" if r in pareto_points else ""
            table.add_row(
                r.config_name, f"{r.accuracy_pct:.2f}",
                f"{r.size_mb:.2f}", f"{r.latency_ms:.2f}",
                f"{r.compression_ratio:.2f}x", style=style,
            )

        console.print(table)
        console.print("[dim]Green rows are Pareto-optimal.[/dim]")

        # LLM final recommendation
        results_dicts = [r.to_dict() for r in valid]
        user_prompt = ANALYZE_USER_TEMPLATE.format(
            model_name=report.model_name,
            original_size_mb=original_size_mb,
            results_json=json.dumps(results_dicts, indent=2),
        )

        try:
            response_text = self.llm.chat(ANALYZE_SYSTEM_PROMPT, user_prompt)
            response = self._parse_json_response(response_text)
            recommendation = response.get("recommendation", "")
            analysis = response.get("analysis", "")

            full_text = ""
            if analysis:
                full_text += f"Analysis:\n{analysis}\n\n"
            if recommendation:
                full_text += f"Recommendation:\n{recommendation}"
                console.print(Panel(
                    f"[bold]Agent Recommendation:[/bold]\n\n{recommendation}",
                    border_style="green",
                ))
            return full_text or "Analysis complete."

        except Exception as e:
            logger.debug(f"Analyst failed: {e}")
            return f"Analyst failed: {e}"

    def _default_configs(self, report: ModelReport) -> list[QuantizationConfig]:
        """Fallback configs if LLM planning fails."""
        configs = []
        applicable = set(report.applicable_methods)

        if "dynamic_int8" in applicable:
            configs.append(QuantizationConfig(method="dynamic_int8"))
        if "static_int8" in applicable:
            configs.append(QuantizationConfig(method="static_int8"))
        if "fp16" in applicable:
            configs.append(QuantizationConfig(method="fp16"))

        return configs if configs else [QuantizationConfig(method="fp16")]

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (fences)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    @staticmethod
    def _print_result(result: EvalResult) -> None:
        """Print a single evaluation result."""
        if result.error:
            console.print(f"  [red]FAILED: {result.error}[/red]")
        else:
            console.print(
                f"  -> Accuracy: {result.accuracy_pct:.2f}%  |  "
                f"Size: {result.size_mb:.2f} MB  |  "
                f"Latency: {result.latency_ms:.2f} ms  |  "
                f"Compression: {result.compression_ratio:.2f}x"
            )
