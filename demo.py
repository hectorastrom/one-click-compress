"""
Interactive Demo — Run the full multi-agent quantization pipeline live in the browser.

Features:
1. Agent Quantization tab: Enter API key, pick model, watch 5 agents discuss
   and decide quantization strategies in real-time with streaming trace
2. Live Captioning tab: Webcam/upload comparison of FP32 vs FP16 vs INT8

Run:
    python demo.py
    python demo.py --port 7861
"""

from __future__ import annotations

import io
import os
import ssl
import time
import json
import copy
import logging
from dataclasses import dataclass
from typing import Generator

import numpy as np
import torch
import gradio as gr
from PIL import Image

# Fix SSL for model downloads
if hasattr(ssl, "_create_unverified_context"):
    ssl._create_default_https_context = ssl._create_unverified_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace rendering helpers
# ---------------------------------------------------------------------------

ROLE_COLORS = {
    "Scanner": "#3498db",
    "Strategist": "#9b59b6",
    "Critic": "#e67e22",
    "Decision Maker": "#2ecc71",
    "Executor": "#34495e",
    "Analyst": "#e74c3c",
}

def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def render_step_html(step_num: int, icon: str, role: str, title: str,
                     content: str, duration_s: float = 0, is_thinking: bool = False) -> str:
    """Render a single trace step as styled HTML."""
    color = ROLE_COLORS.get(role, "#7f8c8d")
    dur = f" · {duration_s:.1f}s" if duration_s > 0 else ""

    if is_thinking:
        return f"""
        <div style="border-left: 3px solid {color}; margin: 10px 0; padding: 10px 14px;
                    background: linear-gradient(90deg, {color}08, transparent);
                    border-radius: 0 8px 8px 0; opacity: 0.7;">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:18px;">{icon}</span>
                <span style="font-weight:700; color:{color}; font-size:13px;">{role}</span>
                <span style="color:#aaa; font-size:11px;">Step {step_num}</span>
                <span style="color:{color}; font-size:12px;">⏳ thinking...</span>
            </div>
        </div>"""

    return f"""
    <div style="border-left: 3px solid {color}; margin: 10px 0; padding: 10px 14px;
                background: linear-gradient(90deg, {color}08, transparent);
                border-radius: 0 8px 8px 0;">
        <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
            <span style="font-size:18px;">{icon}</span>
            <span style="font-weight:700; color:{color}; font-size:13px;">{role}</span>
            <span style="color:#aaa; font-size:11px;">Step {step_num}{dur}</span>
        </div>
        <div style="font-weight:600; font-size:13px; color:#333; margin-bottom:4px;">{_escape(title)}</div>
        <div style="font-size:12px; line-height:1.6; color:#555; white-space:pre-wrap;">{_escape(content)}</div>
    </div>"""


def render_header_html(model_name: str) -> str:
    return f"""
    <div style="font-family: system-ui, sans-serif; max-width:850px; margin:0 auto;">
        <h3 style="text-align:center; margin-bottom:4px;">Multi-Agent Quantization Pipeline</h3>
        <p style="text-align:center; color:#888; font-size:12px; margin-bottom:16px;">
            Model: <strong>{model_name}</strong> — Watch 5 agents collaborate in real-time
        </p>
    """

def render_footer_html(recommendation: str) -> str:
    return f"""
        <div style="margin-top:16px; padding:14px; background:#d4edda;
                    border-radius:8px; border-left:4px solid #28a745;">
            <div style="font-weight:700; color:#155724; margin-bottom:4px;">Final Recommendation</div>
            <div style="color:#155724; font-size:13px; line-height:1.5;">{_escape(recommendation)}</div>
        </div>
    </div>"""


# ---------------------------------------------------------------------------
# Live agent pipeline (generator that yields HTML updates)
# ---------------------------------------------------------------------------

def run_agent_pipeline(api_key: str, model_choice: str) -> Generator[str, None, None]:
    """
    Run the full multi-agent quantization pipeline, yielding HTML after each step.
    This is the core of the demo — the UI updates live as each agent finishes.
    """
    from src.agent import LLMClient, PLAN_SYSTEM_PROMPT, PLAN_USER_TEMPLATE
    from src.agent import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE
    from src.agent import ANALYZE_SYSTEM_PROMPT, ANALYZE_USER_TEMPLATE
    from src.scanner import scan_model
    from src.evaluator import evaluate, measure_model_size_mb
    from src.quantizers.base import build_default_registry, QuantizationConfig
    from src.trace import AgentTrace

    if not api_key or len(api_key) < 10:
        yield "<p style='color:red;'>Please enter a valid OpenAI API key.</p>"
        return

    model_map = {
        "CLIP ViT-B (150M params — fast)": ("openai/clip-vit-base-patch16", "clip-vit-b"),
        "CLIP ViT-L (428M params — impressive)": ("openai/clip-vit-large-patch14", "clip-vit-l"),
    }
    hf_name, short_name = model_map.get(model_choice, ("openai/clip-vit-base-patch16", "clip-vit-b"))

    steps_html = []
    trace = AgentTrace(model_name=short_name)

    def _emit(new_step: str) -> str:
        steps_html.append(new_step)
        return render_header_html(short_name) + "\n".join(steps_html) + "\n</div>"

    # -- Setup --
    llm = LLMClient(provider="openai", api_key=api_key)
    registry = build_default_registry()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # ===== STEP 1: Scanner =====
    yield _emit(render_step_html(1, "🔍", "Scanner", "Analyzing model architecture...", "", is_thinking=True))

    t0 = time.perf_counter()
    from src.clip_wrapper import load_clip_model, get_clip_transform
    model, input_size, _ = load_clip_model(hf_name)
    model.eval().to(device)

    report = scan_model(model, short_name)
    report_dict = report.to_dict()
    sens_summary = report_dict.get("sensitivity_summary", {})
    sens_table = report_dict.get("sensitivity_table", [])
    original_size = measure_model_size_mb(model)
    scan_dur = time.perf_counter() - t0

    top_layers = "\n".join(
        f"  {e['name']}: sensitivity={e['sensitivity']:.3f}, "
        f"Beta(α={e['beta_alpha']:.2f}, β={e['beta_beta']:.2f}), kurtosis={e['kurtosis']:.2f}"
        for e in sens_table[:8]
    )
    scan_text = (
        f"Model: {short_name} ({report.total_params:,} parameters, {original_size:.0f} MB)\n"
        f"Architecture: {report.architecture_hint}\n"
        f"Layers analyzed: {sens_summary.get('total_analyzed_layers', 0)}\n"
        f"Avg sensitivity: {sens_summary.get('avg_sensitivity', 0):.3f}\n"
        f"High-sensitivity layers: {sens_summary.get('high_sensitivity_layers', [])}\n\n"
        f"Most sensitive layers (Beta distribution analysis):\n{top_layers}"
    )
    steps_html.pop()  # remove thinking indicator
    trace.add_step("Scanner", "🔍", "Model Architecture & Beta Distribution Analysis", "", scan_text, scan_dur * 1000)
    yield _emit(render_step_html(1, "🔍", "Scanner", "Model Architecture & Beta Distribution Analysis", scan_text, scan_dur))

    # ===== STEP 2: Baseline =====
    yield _emit(render_step_html(2, "⚡", "Executor", "Evaluating baseline FP32 model...", "", is_thinking=True))

    t0 = time.perf_counter()
    import torchvision
    from torch.utils.data import DataLoader, Subset
    transform = get_clip_transform(input_size)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_ds = Subset(test_ds, list(range(200)))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    baseline = evaluate(model, test_loader, device, original_size, "baseline_fp32", (1, 3, input_size, input_size), latency_runs=10)
    base_dur = time.perf_counter() - t0

    base_text = f"Accuracy: {baseline.accuracy_pct:.2f}%\nSize: {baseline.size_mb:.0f} MB\nLatency: {baseline.latency_ms:.1f} ms"
    steps_html.pop()
    trace.add_step("Executor", "⚡", "Baseline FP32 Evaluation", "", base_text, base_dur * 1000)
    yield _emit(render_step_html(2, "⚡", "Executor", "Baseline FP32 Evaluation", base_text, base_dur))

    # ===== STEP 3: Strategist (LLM call #1) =====
    yield _emit(render_step_html(3, "🧠", "Strategist", "Proposing quantization strategies...", "", is_thinking=True))

    t0 = time.perf_counter()
    plan_prompt = PLAN_USER_TEMPLATE.format(
        model_name=short_name,
        scan_report_json=json.dumps(report_dict, indent=2, default=str),
    )
    try:
        plan_response = llm.chat(PLAN_SYSTEM_PROMPT, plan_prompt)
        plan_parsed = json.loads(_strip_fences(plan_response))
        reasoning = plan_parsed.get("reasoning", "")
        raw_configs = plan_parsed.get("configs", [])
        configs = [QuantizationConfig.from_dict(c) for c in raw_configs]
        # Filter valid
        available = set(registry.available_methods()) & set(report.applicable_methods)
        configs = [c for c in configs if c.method in available]
    except Exception as e:
        reasoning = f"Error: {e}. Using defaults."
        configs = [QuantizationConfig(method="fp16"), QuantizationConfig(method="dynamic_int8")]

    strat_dur = time.perf_counter() - t0
    config_list = "\n".join(f"  • {c.method}" + (f" (per-layer: {len(c.per_layer_config)} rules)" if c.per_layer_config else "") for c in configs)
    strat_text = f"{reasoning}\n\nProposed configs:\n{config_list}"
    steps_html.pop()
    trace.add_step("Strategist", "🧠", "Quantization Strategy Proposals", "", strat_text, strat_dur * 1000, {"configs": [c.to_dict() for c in configs]})
    yield _emit(render_step_html(3, "🧠", "Strategist", "Quantization Strategy Proposals", strat_text, strat_dur))

    # ===== STEP 4: Critic (LLM call #2) =====
    yield _emit(render_step_html(4, "🔎", "Critic", "Reviewing and improving proposals...", "", is_thinking=True))

    t0 = time.perf_counter()
    critic_prompt = CRITIC_USER_TEMPLATE.format(
        model_name=short_name,
        proposed_configs_json=json.dumps([c.to_dict() for c in configs], indent=2),
        sensitivity_summary_json=json.dumps(sens_summary, indent=2),
        top_sensitive_json=json.dumps(sens_table[:15], indent=2, default=str),
    )
    try:
        critic_response = llm.chat(CRITIC_SYSTEM_PROMPT, critic_prompt)
        critic_parsed = json.loads(_strip_fences(critic_response))
        critique = critic_parsed.get("critique", "")
        issues = critic_parsed.get("issues", [])

        # Apply improved mixed_precision
        improved_mp = critic_parsed.get("improved_mixed_precision")
        if improved_mp and isinstance(improved_mp, dict):
            configs = [c for c in configs if c.method != "mixed_precision"]
            configs.append(QuantizationConfig.from_dict(improved_mp))

        # Drop flagged configs
        drop = critic_parsed.get("drop_configs", [])
        if drop:
            configs = [c for c in configs if c.method not in drop]
    except Exception as e:
        critique = f"Critic failed: {e}. Proceeding with original proposals."
        issues = []

    critic_dur = time.perf_counter() - t0
    issues_text = "\n".join(f"  ⚠ {i}" for i in issues) if issues else "  No major issues found."
    critic_text = f"{critique}\n\nIssues:\n{issues_text}\n\nFinal configs: {[c.method for c in configs]}"
    steps_html.pop()
    trace.add_step("Critic", "🔎", "Critical Review & Improvement", "", critic_text, critic_dur * 1000)
    yield _emit(render_step_html(4, "🔎", "Critic", "Critical Review & Improvement", critic_text, critic_dur))

    # ===== STEP 5: Executor — run all configs =====
    all_results = [baseline]
    for i, config in enumerate(configs):
        label = f"Running {config.method}... ({i+1}/{len(configs)})"
        yield _emit(render_step_html(5 + i, "⚡", "Executor", label, "", is_thinking=True))

        t0 = time.perf_counter()
        backend = registry.get(config.method)
        if backend is None:
            steps_html.pop()
            yield _emit(render_step_html(5 + i, "⚡", "Executor", f"{config.method} — SKIPPED", "No backend registered."))
            continue

        quant_result = backend.quantize(model=model, config=config)
        if not quant_result.success:
            exec_dur = time.perf_counter() - t0
            steps_html.pop()
            result_text = f"Quantization failed: {quant_result.error}"
            trace.add_step("Executor", "⚡", f"{config.method} — FAILED", "", result_text, exec_dur * 1000)
            yield _emit(render_step_html(5 + i, "⚡", "Executor", f"{config.method} — FAILED", result_text, exec_dur))
            continue

        eval_result = evaluate(
            quant_result.model, test_loader, device, original_size,
            config.method, (1, 3, input_size, input_size), latency_runs=10,
        )
        all_results.append(eval_result)
        exec_dur = time.perf_counter() - t0

        result_text = (
            f"Accuracy: {eval_result.accuracy_pct:.2f}% (baseline: {baseline.accuracy_pct:.2f}%)\n"
            f"Size: {eval_result.size_mb:.0f} MB → {eval_result.compression_ratio:.2f}x compression\n"
            f"Latency: {eval_result.latency_ms:.1f} ms"
        )
        if config.per_layer_config:
            result_text += f"\nPer-layer rules applied: {len(config.per_layer_config)}"

        steps_html.pop()
        trace.add_step("Executor", "⚡", f"{config.method} — Results", "", result_text, exec_dur * 1000)
        yield _emit(render_step_html(5 + i, "⚡", "Executor", f"{config.method} — Results", result_text, exec_dur))

    # ===== FINAL STEP: Analyst (LLM call #3) =====
    step_n = 5 + len(configs)
    yield _emit(render_step_html(step_n, "📊", "Analyst", "Analyzing results and making recommendation...", "", is_thinking=True))

    t0 = time.perf_counter()
    results_dicts = [r.to_dict() for r in all_results if r.error is None]
    analyze_prompt = ANALYZE_USER_TEMPLATE.format(
        model_name=short_name,
        original_size_mb=original_size,
        results_json=json.dumps(results_dicts, indent=2),
    )
    try:
        analyze_response = llm.chat(ANALYZE_SYSTEM_PROMPT, analyze_prompt)
        analyze_parsed = json.loads(_strip_fences(analyze_response))
        analysis = analyze_parsed.get("analysis", "")
        recommendation = analyze_parsed.get("recommendation", "")
    except Exception as e:
        analysis = f"Analysis failed: {e}"
        recommendation = "Unable to generate recommendation."

    analyst_dur = time.perf_counter() - t0
    analyst_text = f"{analysis}\n\n{recommendation}"
    steps_html.pop()
    trace.add_step("Analyst", "📊", "Final Analysis & Recommendation", "", analyst_text, analyst_dur * 1000)
    trace.final_recommendation = recommendation
    trace.total_duration_ms = sum(s.duration_ms for s in trace.steps)

    # Build final results table
    table_lines = [f"{'Config':<20} {'Accuracy':>10} {'Size':>10} {'Latency':>10} {'Compress':>10}"]
    table_lines.append("-" * 62)
    for r in all_results:
        if r.error is None:
            table_lines.append(f"{r.config_name:<20} {r.accuracy_pct:>9.2f}% {r.size_mb:>8.0f}MB {r.latency_ms:>8.1f}ms {r.compression_ratio:>9.2f}x")

    final_html = render_step_html(step_n, "📊", "Analyst", "Final Analysis & Recommendation", analyst_text, analyst_dur)
    final_html += f"""
    <div style="margin-top:12px; padding:10px; background:#f8f9fa; border-radius:6px;
                font-family:monospace; font-size:11px; white-space:pre;">{chr(10).join(table_lines)}</div>
    """
    final_html += render_footer_html(recommendation)

    # Save trace
    os.makedirs("results", exist_ok=True)
    trace.save("results/agent_trace.json")

    yield _emit(final_html)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text


# ---------------------------------------------------------------------------
# Caption engine (for the live comparison tab)
# ---------------------------------------------------------------------------

@dataclass
class ModelVariant:
    name: str
    model: torch.nn.Module
    size_mb: float
    color: str

caption_engine = None

def get_caption_engine():
    global caption_engine
    if caption_engine is None:
        caption_engine = _build_caption_engine()
    return caption_engine

def _build_caption_engine():
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from torch.quantization import quantize_dynamic

    logger.info("Loading BLIP models for live captioning...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    accel = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    fp32 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").eval().to(accel)
    fp32_size = _model_size(fp32)

    fp16 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).eval().to(accel)
    fp16_size = _model_size(fp16)

    for eng in ("x86", "qnnpack"):
        if eng in torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = eng
            break
    int8 = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").eval()
    int8 = quantize_dynamic(int8, {torch.nn.Linear}, dtype=torch.qint8).eval()
    int8_size = _model_size(int8)

    logger.info("BLIP models ready!")
    return {
        "processor": processor,
        "accel": accel,
        "variants": {
            "FP32 (Original)": ModelVariant("FP32", fp32, fp32_size, "#3498db"),
            "FP16 (Half)": ModelVariant("FP16", fp16, fp16_size, "#2ecc71"),
            "INT8 (Quantized)": ModelVariant("INT8", int8, int8_size, "#e74c3c"),
        },
    }

def _model_size(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / (1024*1024)

def _caption_one(eng, image, vname):
    v = eng["variants"][vname]
    try:
        dev = next(v.model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    inputs = eng["processor"](image, return_tensors="pt")
    inputs = {k: val.to(dev) for k, val in inputs.items()}
    if "FP16" in vname:
        inputs = {k: val.half() if val.is_floating_point() else val for k, val in inputs.items()}
    if dev.type == "mps": torch.mps.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        ids = v.model.generate(**inputs, max_new_tokens=50, num_beams=3)
    if dev.type == "mps": torch.mps.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    return eng["processor"].decode(ids[0], skip_special_tokens=True).strip(), ms

def process_image(image):
    if image is None:
        return "<p>Upload or capture an image first.</p>", None, None
    eng = get_caption_engine()
    pil = Image.fromarray(image).convert("RGB").resize((384, 384))
    results = {}
    for vn in eng["variants"]:
        try:
            cap, ms = _caption_one(eng, pil, vn)
            results[vn] = (cap, ms)
        except Exception as e:
            results[vn] = (f"Error: {e}", 0)

    html = '<div style="font-family:system-ui,sans-serif;">'
    for vn, (cap, ms) in results.items():
        c = eng["variants"][vn].color
        html += f"""<div style="border-left:4px solid {c};padding:10px 14px;margin:6px 0;
            background:linear-gradient(90deg,{c}11,transparent);border-radius:0 8px 8px 0;">
            <div style="font-size:12px;font-weight:700;color:{c};margin-bottom:4px;">{vn} · {ms:.0f}ms</div>
            <div style="font-size:15px;color:#222;">"{_escape(cap)}"</div></div>"""
    html += "</div>"

    # Charts
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    names = list(results.keys())
    lats = [results[n][1] for n in names]
    colors = [eng["variants"][n].color for n in names]

    fig, ax = plt.subplots(figsize=(5, 2))
    bars = ax.barh(names, lats, color=colors, height=0.5)
    for bar, l in zip(bars, lats): ax.text(bar.get_width()+max(lats)*0.02, bar.get_y()+bar.get_height()/2, f"{l:.0f}ms", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(lats)*1.3); ax.invert_yaxis(); ax.set_title("Latency", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); plt.tight_layout()
    buf1 = io.BytesIO(); fig.savefig(buf1, format="png", dpi=100); plt.close(); buf1.seek(0); lat_img = Image.open(buf1)

    sizes = [eng["variants"][n].size_mb for n in names]
    fig, ax = plt.subplots(figsize=(5, 2))
    bars = ax.barh(names, sizes, color=colors, height=0.5)
    for bar, s in zip(bars, sizes): ax.text(bar.get_width()+max(sizes)*0.02, bar.get_y()+bar.get_height()/2, f"{s:.0f}MB", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, max(sizes)*1.3); ax.invert_yaxis(); ax.set_title("Size", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); plt.tight_layout()
    buf2 = io.BytesIO(); fig.savefig(buf2, format="png", dpi=100); plt.close(); buf2.seek(0); size_img = Image.open(buf2)

    return html, lat_img, size_img


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Compression Agent") as app:
        gr.HTML("""
        <h1 style="text-align:center;margin-bottom:4px;">Compression Agent</h1>
        <p style="text-align:center;color:#666;font-size:14px;margin-bottom:16px;">
            Agentic Model Quantization — watch AI agents analyze, debate, and optimize model compression in real-time
        </p>
        """)

        with gr.Tabs():
            # ===== TAB 1: Agent Pipeline (main event) =====
            with gr.Tab("🤖 Agent Quantization", id="agent"):
                gr.HTML("""<p style="color:#888;font-size:13px;margin-bottom:10px;">
                    Enter your OpenAI API key, pick a model, and click <strong>Run Agent Pipeline</strong>
                    to watch 5 AI agents collaborate: Scanner analyzes the model's weight distributions,
                    Strategist proposes quantization configs, Critic reviews and improves them,
                    Executor runs experiments, and Analyst recommends the best approach.
                </p>""")

                with gr.Row():
                    api_key_input = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                        scale=3,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=[
                            "CLIP ViT-B (150M params — fast)",
                            "CLIP ViT-L (428M params — impressive)",
                        ],
                        value="CLIP ViT-B (150M params — fast)",
                        label="Model",
                        scale=2,
                    )
                    run_btn = gr.Button("Run Agent Pipeline", variant="primary", scale=1)

                trace_output = gr.HTML(
                    value="<p style='color:#999;text-align:center;padding:40px;'>Click 'Run Agent Pipeline' to start...</p>",
                )

                run_btn.click(
                    fn=run_agent_pipeline,
                    inputs=[api_key_input, model_dropdown],
                    outputs=[trace_output],
                )

            # ===== TAB 2: Live Captioning =====
            with gr.Tab("📸 Live Captioning", id="caption"):
                gr.HTML("""<p style="color:#888;font-size:13px;margin-bottom:10px;">
                    Capture from webcam or upload an image to compare captions from
                    FP32, FP16, and INT8 quantized BLIP models side-by-side.
                    <em>(Models load on first use — takes ~30s)</em>
                </p>""")

                with gr.Row():
                    with gr.Column(scale=1):
                        webcam = gr.Image(
                            sources=["webcam", "upload"], type="numpy",
                            label="Webcam / Upload", height=380,
                            webcam_options=gr.WebcamOptions(mirror=True),
                        )
                        cap_btn = gr.Button("Generate Captions", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        cap_html = gr.HTML(value="<p style='color:#999;text-align:center;padding:40px;'>Capture an image to see captions...</p>")

                with gr.Row():
                    lat_img = gr.Image(label="Latency", height=220)
                    size_img = gr.Image(label="Model Size", height=220)

                cap_btn.click(fn=process_image, inputs=[webcam], outputs=[cap_html, lat_img, size_img])

    return app


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Bind address: 127.0.0.1 (local only) or 0.0.0.0 (all interfaces)")
    args = parser.parse_args()

    app = build_app()
    print(f"\n  Opening at http://{args.host}:{args.port}")
    print("  (Use --host 0.0.0.0 to allow connections from other devices)\n")
    app.launch(server_name=args.host, server_port=args.port, share=args.share)
