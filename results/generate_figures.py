"""
Generating all result figures for AgentBench-TR README.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent
OUTPUT_DIR  = RESULTS_DIR

sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = {"BM25": "#6366f1", "Dense": "#06b6d4", "Hybrid (RRF)": "#10b981"}


# 1. Retrieval Benchmark 

def plot_retrieval_benchmark():
    """Generating retrieval benchmark bar chart comparing BM25, Dense, and Hybrid."""
    data = {
        "Method":       ["BM25", "Dense", "Hybrid (RRF)", "BM25", "Dense", "Hybrid (RRF)"],
        "Metric":       ["Precision@5"] * 3 + ["Recall@5"] * 3,
        "Score":        [0.260, 0.380, 0.380, 0.800, 0.800, 1.000],
    }
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    fig.suptitle("Retrieval Benchmark: BM25 vs Dense vs Hybrid (RRF)", fontsize=14, fontweight="bold", y=1.02)

    colors = [PALETTE[m] for m in ["BM25", "Dense", "Hybrid (RRF)"]]

    for ax, metric in zip(axes, ["Precision@5", "Recall@5"]):
        subset = df[df["Metric"] == metric]
        bars = ax.bar(subset["Method"], subset["Score"], color=colors, width=0.5, edgecolor="white", linewidth=1.2)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_xlabel("")
        for bar, score in zip(bars, subset["Score"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{score:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / "retrieval_benchmark.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saving retrieval_benchmark.png.. done.")


# 2. Failure Heatmap 

def plot_failure_heatmap():
    """Generating category x failure type heatmap from eval results."""
    data = {
        "Category":        ["conceptual", "factual", "multi-hop"],
        "LOOP_DETECTED":   [2, 1, 0],
        "NO_FAILURE":      [12, 13, 2],
        "TOOL_CALL_FAILED":[3, 1, 1],
    }
    df = pd.DataFrame(data).set_index("Category")

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        annot_kws={"size": 13, "weight": "bold"},
        cbar_kws={"label": "Count"},
    )
    ax.set_title("Failure Type Distribution by Question Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Failure Type", fontsize=11)
    ax.set_ylabel("Category", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    path = OUTPUT_DIR / "failure_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saving failure_heatmap.png.. done.")


#3. Latency Breakdown 

def plot_latency_breakdown():
    """Generating per-agent latency breakdown bar chart."""
    agents   = ["PlannerAgent", "SearchAgent", "ValidatorAgent", "SynthesizerAgent"]
    latency  = [2610, 1850, 3920, 1000]
    colors   = ["#6366f1", "#06b6d4", "#f59e0b", "#10b981"]
    total    = sum(latency)
    pcts     = [v / total * 100 for v in latency]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(agents, latency, color=colors, edgecolor="white", linewidth=1.2, height=0.5)

    for bar, pct, ms in zip(bars, pcts, latency):
        ax.text(bar.get_width() + 40, bar.get_y() + bar.get_height() / 2,
                f"{ms} ms  ({pct:.1f}%)", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_title("Per-Agent Latency Breakdown — Avg Query", fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(latency) * 1.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()

    ax.axvline(x=sum(latency) / len(latency), color="gray", linestyle="--", linewidth=1, label=f"Mean: {sum(latency)//len(latency)} ms")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = OUTPUT_DIR / "latency_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saving latency_breakdown.png.. done.")


#4. Eval Metrics Summary 

def plot_eval_metrics_summary():
    """Generating eval metrics summary panel with key benchmark numbers."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("AgentBench-TR — Eval Metrics Summary (35 Questions)", fontsize=14, fontweight="bold", y=1.02)

    # Consistency
    ax = axes[0]
    value = 0.9049
    ax.pie(
        [value, 1 - value],
        colors=["#10b981", "#e5e7eb"],
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    ax.text(0, 0, f"{value:.2%}", ha="center", va="center", fontsize=18, fontweight="bold", color="#10b981")
    ax.set_title("Consistency Score", fontweight="bold", fontsize=12)

    # Hallucination
    ax = axes[1]
    value = 0.0952
    ax.pie(
        [value, 1 - value],
        colors=["#ef4444", "#e5e7eb"],
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "white"},
    )
    ax.text(0, 0, f"{value:.2%}", ha="center", va="center", fontsize=18, fontweight="bold", color="#ef4444")
    ax.set_title("Hallucination Rate", fontweight="bold", fontsize=12)

    # Failure rate
    ax = axes[2]
    categories = ["NO_FAILURE\n77.14%", "TOOL_CALL\nFAILED\n14.29%", "LOOP\nDETECTED\n8.57%"]
    counts     = [27, 5, 3]
    colors     = ["#10b981", "#f59e0b", "#ef4444"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(count), ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Failure Distribution (35 queries)", fontweight="bold", fontsize=12)
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 32)

    plt.tight_layout()
    path = OUTPUT_DIR / "eval_metrics_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saving eval_metrics_summary.png.. done.")


# Main

if __name__ == "__main__":
    print("Generating all figures...")
    plot_retrieval_benchmark()
    plot_failure_heatmap()
    plot_latency_breakdown()
    plot_eval_metrics_summary()
    print("Generating complete: 4 figures saved to results/.")