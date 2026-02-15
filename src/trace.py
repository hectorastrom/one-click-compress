"""
Agent Trace — capture and replay the multi-agent quantization reasoning process.

Records every step of the agentic pipeline:
- Model scan results
- Each LLM agent call (role, prompt, response)
- Quantization decisions and outcomes
- Final recommendations

Traces are saved as JSON for caching and replay.
"""

from __future__ import annotations

import json
import os
import time
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TraceStep:
    """A single step in the agent trace."""
    step_number: int
    agent_role: str          # e.g. "Scanner", "Strategist", "Critic", "Executor", "Analyst"
    agent_icon: str          # emoji for display
    title: str               # short description
    input_summary: str       # what was given to this agent
    output_text: str         # the agent's response/output
    duration_ms: float = 0   # how long this step took
    timestamp: str = ""      # ISO timestamp
    data: dict[str, Any] = field(default_factory=dict)  # structured data (configs, results, etc.)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTrace:
    """Complete trace of an agentic quantization session."""
    model_name: str
    created_at: str = ""
    total_duration_ms: float = 0
    steps: list[TraceStep] = field(default_factory=list)
    final_recommendation: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def add_step(
        self,
        agent_role: str,
        agent_icon: str,
        title: str,
        input_summary: str,
        output_text: str,
        duration_ms: float = 0,
        data: dict[str, Any] | None = None,
    ) -> TraceStep:
        """Add a step to the trace."""
        step = TraceStep(
            step_number=len(self.steps) + 1,
            agent_role=agent_role,
            agent_icon=agent_icon,
            title=title,
            input_summary=input_summary,
            output_text=output_text,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            data=data or {},
        )
        self.steps.append(step)
        logger.info(f"Trace step {step.step_number}: [{agent_role}] {title}")
        return step

    def save(self, path: str = "results/agent_trace.json") -> str:
        """Save trace to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "model_name": self.model_name,
            "created_at": self.created_at,
            "total_duration_ms": self.total_duration_ms,
            "final_recommendation": self.final_recommendation,
            "steps": [s.to_dict() for s in self.steps],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Trace saved to {path}")
        return path

    @classmethod
    def load(cls, path: str) -> AgentTrace:
        """Load a trace from JSON file."""
        with open(path) as f:
            data = json.load(f)
        trace = cls(
            model_name=data["model_name"],
            created_at=data.get("created_at", ""),
            total_duration_ms=data.get("total_duration_ms", 0),
            final_recommendation=data.get("final_recommendation", ""),
        )
        for s in data.get("steps", []):
            trace.steps.append(TraceStep(**s))
        return trace

    def to_html(self) -> str:
        """Render the trace as a styled HTML timeline for display in Gradio."""
        html = []
        html.append(f"""
        <div style="font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto;">
            <h2 style="text-align: center; margin-bottom: 4px;">Agent Reasoning Trace</h2>
            <p style="text-align: center; color: #888; font-size: 13px; margin-bottom: 24px;">
                Model: <strong>{self.model_name}</strong> &middot;
                {len(self.steps)} steps &middot;
                {self.total_duration_ms/1000:.1f}s total
            </p>
        """)

        role_colors = {
            "Scanner": "#3498db",
            "Strategist": "#9b59b6",
            "Critic": "#e67e22",
            "Decision Maker": "#2ecc71",
            "Executor": "#34495e",
            "Analyst": "#e74c3c",
        }

        for step in self.steps:
            color = role_colors.get(step.agent_role, "#7f8c8d")
            duration_str = f"{step.duration_ms/1000:.1f}s" if step.duration_ms > 0 else ""

            html.append(f"""
            <div style="border-left: 3px solid {color}; margin: 12px 0; padding: 12px 16px;
                        background: linear-gradient(90deg, {color}08, transparent);
                        border-radius: 0 8px 8px 0;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 20px;">{step.agent_icon}</span>
                    <span style="font-weight: 700; color: {color}; font-size: 14px;">
                        {step.agent_role}
                    </span>
                    <span style="color: #aaa; font-size: 12px;">
                        Step {step.step_number} {('&middot; ' + duration_str) if duration_str else ''}
                    </span>
                </div>
                <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px; color: #333;">
                    {step.title}
                </div>
            """)

            # Input summary (collapsible)
            if step.input_summary:
                html.append(f"""
                <details style="margin-bottom: 8px;">
                    <summary style="cursor: pointer; font-size: 12px; color: #888;">
                        Show input context
                    </summary>
                    <div style="font-size: 12px; color: #666; padding: 8px; margin-top: 4px;
                                background: #f8f9fa; border-radius: 4px; white-space: pre-wrap;
                                max-height: 200px; overflow-y: auto;">
{_escape_html(step.input_summary[:2000])}
                    </div>
                </details>
                """)

            # Output (main content)
            html.append(f"""
                <div style="font-size: 13px; line-height: 1.6; color: #444;
                            padding: 8px 0; white-space: pre-wrap;">
{_escape_html(step.output_text[:3000])}
                </div>
            """)

            # Data badges
            if step.data:
                if "configs" in step.data:
                    configs = step.data["configs"]
                    if isinstance(configs, list):
                        badges = " ".join(
                            f'<span style="display:inline-block; padding:2px 8px; margin:2px; '
                            f'background:{color}22; color:{color}; border-radius:12px; font-size:11px;">'
                            f'{c.get("method", c) if isinstance(c, dict) else c}</span>'
                            for c in configs
                        )
                        html.append(f'<div style="margin-top: 4px;">{badges}</div>')

                if "results_table" in step.data:
                    html.append(f"""
                    <div style="margin-top: 8px; font-size: 12px; font-family: monospace;
                                background: #f8f9fa; padding: 8px; border-radius: 4px;
                                max-height: 200px; overflow-y: auto;">
{_escape_html(step.data["results_table"])}
                    </div>
                    """)

            html.append("</div>")  # close step div

        # Final recommendation
        if self.final_recommendation:
            html.append(f"""
            <div style="margin-top: 20px; padding: 16px; background: #d4edda;
                        border-radius: 8px; border-left: 4px solid #28a745;">
                <div style="font-weight: 700; color: #155724; margin-bottom: 6px;">
                    Final Recommendation
                </div>
                <div style="color: #155724; font-size: 14px; line-height: 1.5;">
                    {_escape_html(self.final_recommendation)}
                </div>
            </div>
            """)

        html.append("</div>")
        return "\n".join(html)


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
