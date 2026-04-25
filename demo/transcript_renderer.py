"""renders an EpisodeResult (or two side-by-side) to markdown for the demo dir.

three primary entrypoints:
- render_transcript(result)               -> single-column markdown
- render_side_by_side(left, right, ...)   -> two-column "baseline | trained"
- render_demo_set(out_dir, results_by_label) -> writes one .md per (case, models)

design choices:
- citizen utterances appear as block quotes; advisor actions appear as labeled
  bullets with the JSON-ish action body. keeps the diff between baseline and
  trained legible at a glance.
- final ActionPlan rendered as a structured section, not raw JSON. shows scheme
  ids + framework ids + dlsa contact prominently.
- truncation: a per-action 240-char cap keeps lines readable; full content is
  always available in the EpisodeResult itself.
"""

from __future__ import annotations

from pathlib import Path

from training.rollout import EpisodeResult, TurnLog


def _truncate(s: str, n: int = 240) -> str:
    s = s.strip()
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _render_action(action) -> str:
    t = action.type
    if t == "ASK":
        return f"**ASK** ({action.language}): {_truncate(action.question)}"
    if t == "PROBE":
        return (
            f"**PROBE** ({action.sensitive_topic}, {action.language}): {_truncate(action.question)}"
        )
    if t == "EXPLAIN":
        return (
            f"**EXPLAIN** (literacy={action.target_literacy}, {action.language}): "
            f"{_truncate(action.content)}"
        )
    if t == "FINALIZE":
        return "**FINALIZE** — see plan below."
    return f"**{t}** {action!r}"


def _render_plan(plan) -> str:
    if plan is None:
        return "_(no plan submitted)_"
    lines: list[str] = []
    if plan.schemes:
        lines.append("**Schemes**")
        for s in plan.schemes:
            lines.append(
                f"- `{s.scheme_id}` — rationale: {', '.join(s.rationale_facts) or '(none)'}"
            )
    else:
        lines.append("_(no schemes)_")
    if plan.legal_routes:
        lines.append("")
        lines.append("**Legal routes**")
        for r in plan.legal_routes:
            contact = r.free_legal_aid_contact
            lines.append(
                f"- `{r.framework_id}` → forum: {r.forum} → {contact.authority} `{contact.contact_id}`"
            )
    else:
        lines.append("_(no legal routes)_")
    lines.append("")
    lines.append(f"**Most important next step**: {_truncate(plan.most_important_next_step)}")
    summary = plan.plain_summary
    lines.append("")
    lines.append(f"_Summary ({summary.language}): {_truncate(summary.text)}_")
    return "\n".join(lines)


def _render_turn(turn: TurnLog) -> list[str]:
    out: list[str] = []
    obs = turn.observation_in
    if obs and obs.citizen_utterance:
        out.append(f"> Citizen: {_truncate(obs.citizen_utterance)}")
    out.append(f"- {_render_action(turn.action)}")
    return out


def render_transcript(result: EpisodeResult, *, title: str | None = None) -> str:
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append(
        f"**seed**: {result.seed} · **difficulty**: {result.difficulty or '-'} · "
        f"**finalized**: {'✓' if result.finalized else '✗'} · "
        f"**total reward**: {result.total_reward:.3f} · "
        f"**turns**: {len(result.turns)} · **sim_leak**: {result.sim_leak_count}"
    )
    if result.error:
        lines.append("")
        lines.append(f"**error**: {result.error}")
    lines.append("")
    lines.append("## Conversation")
    for turn in result.turns:
        lines.extend(_render_turn(turn))
    lines.append("")
    lines.append("## Action plan")
    final_plan = None
    if result.turns and result.turns[-1].action.type == "FINALIZE":
        final_plan = result.turns[-1].action.plan
    lines.append(_render_plan(final_plan))
    lines.append("")
    lines.append("## Reward breakdown")
    if result.final_breakdown:
        lines.append("| component | value |")
        lines.append("|---|---|")
        for k in sorted(result.final_breakdown.keys()):
            v = result.final_breakdown[k]
            lines.append(f"| `{k}` | {v:.3f} |")
    else:
        lines.append("_(no breakdown)_")
    return "\n".join(lines)


def render_side_by_side(
    left: EpisodeResult,
    right: EpisodeResult,
    *,
    left_label: str = "baseline",
    right_label: str = "trained",
    title: str | None = None,
) -> str:
    """two-column markdown. since github md doesn't render fenced two-column
    layouts cleanly, we emit a header table summarizing both, then each column
    as a sub-section. easy to read in a code editor or rendered."""
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    lines.append("| | total | finalized | turns | gates passed |")
    lines.append("|---|---|---|---|---|")
    for label, r in ((left_label, left), (right_label, right)):
        gates_passed = (
            r.final_breakdown.get("gate_format_violation", 0) == 0
            and r.final_breakdown.get("gate_hallucination", 0) == 0
            and r.final_breakdown.get("gate_contradiction", 0) == 0
        )
        lines.append(
            f"| **{label}** | {r.total_reward:.3f} | "
            f"{'✓' if r.finalized else '✗'} | {len(r.turns)} | "
            f"{'✓' if gates_passed else '✗'} |"
        )
    lines.append("")
    lines.append(f"## {left_label}")
    lines.append(render_transcript(left))
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"## {right_label}")
    lines.append(render_transcript(right))
    return "\n".join(lines)


def render_demo_set(
    out_dir: Path,
    results_by_label: dict[str, list[EpisodeResult]],
    *,
    cases: list[str] | None = None,
) -> dict[str, Path]:
    """writes one markdown file per case, side-by-side across labels.
    results_by_label maps "baseline" / "prompted" / "trained" -> list[EpisodeResult]
    keyed by index. cases (optional) names each index for the file name.
    returns a mapping case_name -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not results_by_label:
        return {}
    labels = list(results_by_label.keys())
    n = max(len(r) for r in results_by_label.values())
    paths: dict[str, Path] = {}
    for i in range(n):
        case_name = cases[i] if cases and i < len(cases) else f"case_{i + 1:03d}"
        # for now we render baseline-vs-trained pairs across the first two labels.
        if len(labels) < 2:
            r = results_by_label[labels[0]][i]
            md = render_transcript(r, title=case_name)
        else:
            left = results_by_label[labels[0]][i]
            right = results_by_label[labels[1]][i]
            md = render_side_by_side(
                left,
                right,
                left_label=labels[0],
                right_label=labels[1],
                title=case_name,
            )
        path = out_dir / f"{case_name}.md"
        path.write_text(md, encoding="utf-8")
        paths[case_name] = path
    return paths


__all__ = [
    "render_demo_set",
    "render_side_by_side",
    "render_transcript",
]
