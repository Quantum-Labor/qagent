"""QAgent HuggingFace Space - QAOA tool selection, runs on CPU Basic.

The Space serves *precomputed* QAOA results (p=4, 160 steps, seed 0 - the
documented v0.2 config) for the 50 qagent-mini tasks, so the experience is
instant and needs no torch / pennylane in the image. The pure-Python classical
solvers (brute-force exact, greedy top-k) run *live* on demand so visitors can
confirm the served numbers are real. There is no IBM hardware path and no OAuth:
the Space is fully open.

Design (see README.md): a hero with the QAOA ansatz, a task explorer with a 4x4
tool grid that highlights which tools each solver picks, score cards with
approximation ratios, a score-landscape chart showing where QAOA / greedy /
optimal fall among all size-k subsets, an exploration history, and the
qagent-mini-50 leaderboard.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safety import RateLimiter, default_persist_path

from qagent.oracle.bruteforce import brute_force_best
from qagent.oracle.greedy import greedy_top_k
from qagent.qaoa.encoding import ToolScoring

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "precomputed" / "benchmark_results.json"
ASSETS = HERE / "assets"
REPO_URL = "https://github.com/Quantum-Labor/qagent"

_RATE_LIMITER = RateLimiter(window_seconds=3, daily_cap=5000, persist_path=default_persist_path())


def _load_results() -> dict[str, Any]:
    try:
        return json.loads(RESULTS_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    except (FileNotFoundError, json.JSONDecodeError):
        return {"config": {}, "summary": {}, "tasks": []}


RESULTS = _load_results()
TASKS: list[dict[str, Any]] = RESULTS.get("tasks", [])
TASK_BY_ID: dict[str, dict[str, Any]] = {t["id"]: t for t in TASKS}

# The source dataset (weights + full synergy matrix) is bundled so the live
# brute-force / greedy check reconstructs the exact scoring behind each task.
_DATASET_PATH = HERE / "precomputed" / "dataset.json"


def _load_dataset() -> dict[str, dict[str, Any]]:
    try:
        rows = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
        return {r["id"]: r for r in rows}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


DATASET_BY_ID = _load_dataset()


def _read_asset(name: str) -> str:
    try:
        return (ASSETS / name).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _client_ip(request: gr.Request | None) -> str:
    if request is None:
        return "unknown"
    fwd = request.headers.get("x-forwarded-for") if request.headers else None
    if fwd:
        return fwd.split(",")[0].strip()
    return getattr(request.client, "host", "unknown") or "unknown"


# --- rendering helpers (pure, unit-testable) --------------------------------


def task_label(task: dict[str, Any]) -> str:
    return f"{task['id']} · {task['tier']} · N={task['n_tools']} k={task['k']}"


def task_choices() -> list[str]:
    return [task_label(t) for t in TASKS]


def _label_to_id(label: str) -> str:
    return label.split(" ", 1)[0]


def render_tool_grid(task: dict[str, Any]) -> str:
    optimal = set(task["optimal"]["subset"])
    qaoa = set(task["qaoa"]["subset"])
    greedy = set(task["greedy"]["subset"])
    cards: list[str] = []
    for tool in task["tools"]:
        i = tool["index"]
        w = tool["weight"]
        classes = "qa-tool-card"
        if i in optimal:
            classes += " sel-optimal"
        if i in qaoa:
            classes += " sel-qaoa"
        tags = []
        if i in optimal:
            tags.append('<span class="qa-tag optimal">optimal</span>')
        if i in qaoa:
            tags.append('<span class="qa-tag qaoa">qaoa</span>')
        if i in greedy:
            tags.append('<span class="qa-tag greedy">greedy</span>')
        tags_html = f'<div class="qa-tags">{"".join(tags)}</div>' if tags else ""
        bar = f'<div class="qa-tool-bar"><span style="width:{min(w, 1.0) * 100:.0f}%"></span></div>'
        cards.append(
            f'<div class="{classes}"><div class="qa-tool-name">{tool["name"]}</div>'
            f'<div class="qa-tool-weight">w = {w:.3f}</div>{bar}{tags_html}</div>'
        )
    legend = (
        '<div class="qa-legend">'
        '<span><span class="dot" style="background:#34D399"></span>optimal (brute-force)</span>'
        '<span><span class="dot" style="background:#67E8F9"></span>QAOA</span>'
        '<span><span class="dot" style="background:#F59E0B"></span>greedy</span>'
        "</div>"
    )
    return f'<div class="qa-tool-grid">{"".join(cards)}</div>{legend}'


def render_scores(task: dict[str, Any]) -> str:
    opt = task["optimal"]
    qa = task["qaoa"]
    gd = task["greedy"]
    exact = "exact match" if qa["exact_match"] else "approximate"

    def card(kind: str, title: str, score: float, sub: str) -> str:
        return (
            f'<div class="qa-score-card {kind}"><h4>{title}</h4>'
            f'<div class="qa-score-val">{score:.3f}</div>'
            f'<div class="qa-score-sub">{sub}</div></div>'
        )

    return (
        '<div class="qa-scores">'
        + card(
            "optimal",
            "Optimal (brute-force)",
            opt["score"],
            f"{task['k']} of {task['n_tools']} tools",
        )
        + card("qaoa", "QAOA", qa["score"], f"ratio {qa['approx_ratio']:.3f} · {exact}")
        + card("greedy", "Greedy top-k", gd["score"], f"ratio {gd['approx_ratio']:.3f}")
        + "</div>"
    )


def landscape_figure(task: dict[str, Any]) -> Any:
    hist = task["score_hist"]
    edges = hist["bin_edges"]
    counts = hist["counts"]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    width = (edges[1] - edges[0]) * 0.9 if len(edges) > 1 else 0.1

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.6, 3.3), dpi=110)
    fig.patch.set_facecolor("#0B0B16")
    ax.set_facecolor("#0B0B16")
    ax.bar(
        centers, counts, width=width, color="#3A3460", edgecolor="none", label="all size-k subsets"
    )
    for score, color, label in [
        (task["greedy"]["score"], "#F59E0B", "greedy"),
        (task["qaoa"]["score"], "#22D3EE", "QAOA"),
        (task["optimal"]["score"], "#34D399", "optimal"),
    ]:
        ax.axvline(score, color=color, linewidth=2.2, label=label)
    ax.set_xlabel("subset score", color="#9CA3AF", fontsize=10)
    ax.set_ylabel(f"# subsets ({hist['n_subsets']})", color="#9CA3AF", fontsize=10)
    ax.set_title("Score landscape: where each solver lands", color="#E5E7EB", fontsize=12, pad=10)
    ax.tick_params(colors="#6B6788", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2A2440")
    ax.legend(
        facecolor="#15152A", edgecolor="#2A2440", labelcolor="#E5E7EB", fontsize=9, loc="upper left"
    )
    fig.tight_layout()
    return fig


def history_figure(history: list[dict[str, Any]]) -> Any:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7.6, 2.8), dpi=110)
    fig.patch.set_facecolor("#0B0B16")
    ax.set_facecolor("#0B0B16")
    if history:
        xs = list(range(len(history)))
        ax.plot(xs, [h["qaoa"] for h in history], "-o", color="#22D3EE", label="QAOA", linewidth=2)
        ax.plot(
            xs, [h["greedy"] for h in history], "-o", color="#F59E0B", label="greedy", linewidth=2
        )
        ax.axhline(1.0, color="#34D399", linewidth=1.2, linestyle="--", label="optimal")
        ax.set_xticks(xs)
        ax.set_xticklabels([h["id"] for h in history], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.5, 1.05)
        ax.legend(
            facecolor="#15152A",
            edgecolor="#2A2440",
            labelcolor="#E5E7EB",
            fontsize=9,
            loc="lower left",
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Explore tasks to build a history",
            ha="center",
            va="center",
            color="#6B6788",
            fontsize=11,
            transform=ax.transAxes,
        )
    ax.set_ylabel("approx ratio", color="#9CA3AF", fontsize=10)
    ax.set_title("Approximation-ratio history (this session)", color="#E5E7EB", fontsize=12, pad=8)
    ax.tick_params(colors="#6B6788", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2A2440")
    fig.tight_layout()
    return fig


def _task_description(task: dict[str, Any]) -> str:
    return (
        f"**{task['id']}** · {task['tier']} pool · select **{task['k']}** of "
        f"**{task['n_tools']}** tools. {task['description']}"
    )


def select_task(
    label: str, history: list[dict[str, Any]]
) -> tuple[str, str, str, Any, Any, list[dict[str, Any]]]:
    task = TASK_BY_ID[_label_to_id(label)]
    history = [h for h in history if h["id"] != task["id"]]
    history.append(
        {
            "id": task["id"],
            "qaoa": task["qaoa"]["approx_ratio"],
            "greedy": task["greedy"]["approx_ratio"],
        }
    )
    history = history[-12:]
    return (
        _task_description(task),
        render_tool_grid(task),
        render_scores(task),
        landscape_figure(task),
        history_figure(history),
        history,
    )


def verify_live(label: str, request: gr.Request | None = None) -> str:
    """Run brute-force + greedy live to prove the served numbers are real."""
    verdict = _RATE_LIMITER.check_and_register(ip=_client_ip(request), now=datetime.now(UTC))
    if not verdict.allowed:
        return f"_{verdict.detail}_"
    task = TASK_BY_ID[_label_to_id(label)]
    entry = DATASET_BY_ID.get(task["id"])
    if entry is None:
        return "_Source scoring unavailable for live verification._"
    sc = ToolScoring(
        weights=tuple(entry["weights"]),
        synergy=tuple(tuple(row) for row in entry["synergy"]),
    )
    k = task["k"]
    bf_sub, bf_score = brute_force_best(sc, k)
    gd_sub, gd_score = greedy_top_k(sc, k)
    opt = task["optimal"]["score"]
    bf_ok = abs(bf_score - opt) < 1e-6
    return (
        f"Live solve of **{task['id']}** (pure-Python, no precomputed lookup):\n\n"
        f"- brute-force optimum: `{sorted(bf_sub)}` score **{bf_score:.4f}** "
        f"{'matches' if bf_ok else 'DIFFERS FROM'} the served optimal `{opt:.4f}`\n"
        f"- greedy top-k: `{sorted(gd_sub)}` score **{gd_score:.4f}**\n\n"
        f"The served QAOA result (score {task['qaoa']['score']:.4f}, ratio "
        f"{task['qaoa']['approx_ratio']:.3f}) was precomputed with p={RESULTS['config'].get('p')} "
        f"on the PennyLane simulator."
    )


def leaderboard_markdown() -> str:
    s = RESULTS.get("summary", {})
    if not s:
        return "_Leaderboard unavailable._"

    def cell(exact: int, ratio: float, n: int) -> str:
        return f"{100 * exact // n}% ({exact}/{n}) · {ratio:.3f}"

    def row(group: str, g: dict[str, Any]) -> str:
        n = g["n"]
        gd = cell(g["greedy_exact"], g["greedy_ratio"], n)
        qa = cell(g["qaoa_exact"], g["qaoa_ratio"], n)
        return f"| {group} | 100% ({n}/{n}) | {gd} | {qa} |"

    lines = [
        "| Group | Brute-force | Greedy | QAOA |",
        "| --- | --- | --- | --- |",
    ]
    for key, name in [("small", "small (N=8, k=3)"), ("full", "full (N=16, k=5)"), ("all", "all")]:
        if key in s:
            lines.append(row(name, s[key]))
    cfg = RESULTS.get("config", {})
    note = (
        "\n\nMetric: exact-match (and mean approximation ratio). QAOA config: "
        f"p={cfg.get('p')}, {cfg.get('steps')} steps, {cfg.get('shots')} shots, "
        f"seed {cfg.get('seed')}, x mixer. Brute-force is the oracle (100% by construction)."
    )
    return "\n".join(lines) + note


# --- UI ---------------------------------------------------------------------

_THEME = gr.themes.Base(
    primary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#0B0B16",
    body_text_color="#E5E7EB",
    background_fill_primary="#15152A",
    background_fill_secondary="#0B0B16",
    border_color_primary="#2A2440",
    button_primary_background_fill="#7C3AED",
    button_primary_background_fill_hover="#8B5CF6",
    button_primary_text_color="#FFFFFF",
    block_background_fill="#15152A",
    block_border_color="#2A2440",
    input_background_fill="#15152A",
)


def _hero_html() -> str:
    hero_svg = _read_asset("hero_qaoa.svg")
    art = (
        f'<div class="qa-hero-art">{hero_svg}</div>'
        if hero_svg
        else '<div class="qa-hero-art"></div>'
    )
    return f"""
<div class="qa-hero"><div class="qa-hero-inner">
  {art}
  <div class="qa-hero-copy">
    <h1 class="qa-title">QAgent</h1>
    <p class="qa-tagline">Quantum-optimized tool selection for LLM agents.
    Pick the best subset of <b>k</b> tools from <b>N</b> with QAOA.</p>
    <p class="qa-sub">Tools interact - some synergise, some are redundant - so the
    optimal set is a combinatorial optimization, not a ranking. QAgent encodes
    "select k of N" on one qubit per tool and solves it with the Quantum
    Approximate Optimization Algorithm.</p>
    <div class="qa-badges">
      <span class="qa-badge">v0.2.0</span>
      <a class="qa-badge" href="{REPO_URL}">GitHub</a>
      <span class="qa-badge cyan">Project 2 of 3</span>
      <span class="qa-badge green">simulator</span>
      <span class="qa-badge amber">Apache-2.0</span>
    </div>
  </div>
</div></div>
"""


_WHAT = """
### What is this?

An LLM agent with many tools should only be handed a few per task - to fit a
context budget, cut cost, and reduce confusion. Picking the *k* individually most
relevant tools is greedy and often wrong, because tools have **pairwise
interactions**: a retriever plus a summariser synergise; two overlapping search
tools are redundant. The value of a tool set is

```
score(S) = sum of tool relevances + sum of pairwise interactions
```

and the task is to maximise it subject to choosing exactly *k* tools - a quadratic
binary optimization, exactly the structure QAOA targets.
"""

_HOW = """
### How it works (in 30 seconds)

- **Encode** the task as per-tool relevances and a pairwise interaction matrix.
- **Map** "select k of N" to N qubits (one-hot: qubit i = tool i in/out), folding
  the cardinality constraint into the cost Hamiltonian as a penalty.
- **Optimize** a QAOA ansatz (cost layer + mixer) with PyTorch on the PennyLane
  simulator; measure and keep the best valid subset.
- **Compare** against an exact brute-force oracle and a greedy top-k baseline.

This Space serves results precomputed with that pipeline; the classical solvers
run live so you can check the numbers.
"""

_ABOUT = f"""
### About

QAgent is project 2 of 3 in the Quantum Co-Processor program, after
[QVerify](https://github.com/Quantum-Labor/qverify) (quantum-assisted reasoning
verification) and alongside
[QRoute](https://github.com/Quantum-Labor/qroute) (a VQC mixture-of-experts
router). Source and full benchmarks: [{REPO_URL}]({REPO_URL}).

**Honest scope.** Simulator only; no quantum advantage is claimed. On
qagent-mini-50, QAOA matches the optimum on every N=8 task and reaches a 0.915
mean approximation ratio at N=16, but exact-match at N=16 remains hard (the
documented next step is Dicke-initialised XY-QAOA). The point is the architecture
and an honest baseline.
"""


def build_demo() -> gr.Blocks:
    # In Gradio 6.x, theme and css are passed to launch(), not the Blocks
    # constructor (where they are silently ignored) -- mirrors the QVerify Space.
    with gr.Blocks(title="QAgent - quantum tool selection") as demo:
        gr.HTML(_hero_html())
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(_WHAT)
            with gr.Column(scale=1):
                gr.Markdown(_HOW)

        gr.HTML('<div class="qa-section-title">Try it now</div>')
        history = gr.State([])
        default_label = task_choices()[0] if TASKS else None
        task_dd = gr.Dropdown(
            choices=task_choices(),
            value=default_label,
            label="qagent-mini-50 task",
            info="Pick a task to see which tools each solver selects.",
        )
        task_desc = gr.Markdown()
        grid = gr.HTML()
        scores = gr.HTML()
        with gr.Row():
            verify_btn = gr.Button("Verify live (brute-force + greedy)", variant="primary")
        verify_out = gr.Markdown()
        with gr.Row():
            landscape = gr.Plot(label="Score landscape")
            history_plot = gr.Plot(label="History")

        gr.HTML('<div class="qa-section-title">qagent-mini-50 leaderboard</div>')
        gr.Markdown(leaderboard_markdown())

        gr.Markdown(_ABOUT)

        outputs = [task_desc, grid, scores, landscape, history_plot, history]
        task_dd.change(select_task, inputs=[task_dd, history], outputs=outputs)
        verify_btn.click(verify_live, inputs=[task_dd], outputs=[verify_out])
        if default_label is not None:
            demo.load(select_task, inputs=[task_dd, history], outputs=outputs)
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        theme=_THEME,
        css=_read_asset("styles.css"),
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
