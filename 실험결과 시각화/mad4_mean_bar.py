import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent.parent / "stats" / "mad4_stats_all.json"
with open(DATA_PATH, encoding="utf-8") as f:
    raw = json.load(f)

MODEL_LABELS = {
    "judge_results/exp04_mad3/gemma": "Gemma3-4B",
    "judge_results/exp04_mad3/qwen":  "Qwen3.5-9B",
    "judge_results/exp04_mad3/llama": "Llama3-8B",
    "judge_results/exp04_mad3/gpt":   "GPT4o-mini",
}
MODEL_COLORS = {
    "Gemma3-4B":  "#4C72B0",
    "Qwen3.5-9B": "#55A868",
    "Llama3-8B":  "#C44E52",
    "GPT4o-mini": "#DD8452",
}

SCORE_KEYS = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]
CAT_LABELS = ["Domain Match", "Score-Rationale\nConsistency", "Specificity", "Groundedness"]

models = [MODEL_LABELS[k] for k in MODEL_LABELS]
data   = {MODEL_LABELS[k]: raw[k] for k in MODEL_LABELS if k in raw}

# ── 레이아웃 ──────────────────────────────────────────────────────────────────
x         = np.arange(len(SCORE_KEYS))
bar_width = 0.18
offsets   = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * bar_width

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#F8F9FA")
ax.set_facecolor("#F8F9FA")

# ── 막대 그리기 ───────────────────────────────────────────────────────────────
for i, model in enumerate(models):
    values = [data[model][key] for key in SCORE_KEYS]
    bars = ax.bar(
        x + offsets[i], values,
        width=bar_width,
        color=MODEL_COLORS[model],
        alpha=0.85,
        label=model,
        zorder=3,
    )
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=7.5, color="#333333",
        )

# ── 축·스타일 ─────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(CAT_LABELS, fontsize=12)
ax.set_ylabel("Mean Score", fontsize=12)
ax.set_ylim(0, 5.2)
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

ax.text(0.01, 0.97, "↑ higher is better",
        transform=ax.transAxes, fontsize=9, color="#888888", va="top")

ax.legend(fontsize=9.5, frameon=True, framealpha=0.9, loc="upper right")
ax.set_title("Mean Score by Model & Category", fontsize=14, fontweight="bold", pad=10)
fig.suptitle("MAD4 (Strict/Lenient) – Model Comparison",
             fontsize=16, fontweight="bold", y=1.02)

plt.tight_layout()
out = Path(__file__).parent / "mad4_mean_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"저장 완료: {out}")
