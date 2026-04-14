import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
with open(BASE / "stats" / "judge_single_all.json",  encoding="utf-8") as f: single_raw = json.load(f)
with open(BASE / "stats" / "judge_MAD3_all.json",    encoding="utf-8") as f: mad1_raw   = json.load(f)
with open(BASE / "stats" / "mad4_stats_all.json",    encoding="utf-8") as f: mad2_raw   = json.load(f)

# ── 모델 매핑 ─────────────────────────────────────────────────────────────────
MODELS = ["GPT4o-mini", "Qwen3.5-9B", "Gemma3-4B", "Llama3-8B"]
MODEL_COLORS = {
    "Gemma3-4B":  "#4C72B0",
    "Qwen3.5-9B": "#55A868",
    "Llama3-8B":  "#C44E52",
    "GPT4o-mini": "#DD8452",
}

SINGLE_KEYS = ["judge_single_results_gpt", "judge_single_results_qwen",
               "judge_single_results_gemma", "judge_single_results_llama"]
MAD1_KEYS   = ["mad3_results_gpt", "mad3_results_qwen",
               "mad3_results_gemma", "mad3_results_llama"]
MAD2_KEYS   = ["mad4_results_gpt", "mad4_results_qwen",
               "mad4_results_gemma", "mad4_results_llama"]

SCORE_KEYS = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]
CAT_LABELS = ["Domain\nMatch", "Score-Rationale\nConsistency", "Specificity", "Groundedness"]

# ── 데이터 정규화 (모델 × 카테고리 mean 행렬) ──────────────────────────────
def extract_single_mad1(raw, keys):
    """{ model_label: { score_key: mean } }"""
    result = {}
    for label, key in zip(MODELS, keys):
        result[label] = {
            sk: raw[key]["judge_stats"][sk]["mean"]
            for sk in SCORE_KEYS
        }
    return result

def extract_mad2(raw, keys):
    result = {}
    for label, key in zip(MODELS, keys):
        result[label] = {sk: raw[key][sk] for sk in SCORE_KEYS}
    return result

data_single = extract_single_mad1(single_raw, SINGLE_KEYS)
data_mad1   = extract_single_mad1(mad1_raw,   MAD1_KEYS)
data_mad2   = extract_mad2(mad2_raw,           MAD2_KEYS)

DATASETS = [
    ("Judge Single",  data_single),
    ("MAD1",          data_mad1),
    ("MAD2",          data_mad2),
]

# ── 레이아웃 ──────────────────────────────────────────────────────────────────
x         = np.arange(len(SCORE_KEYS))
bar_width = 0.18
offsets   = np.linspace(-(len(MODELS)-1)/2, (len(MODELS)-1)/2, len(MODELS)) * bar_width

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
fig.patch.set_facecolor("#F8F9FA")

for ax, (title, data) in zip(axes, DATASETS):
    ax.set_facecolor("#F8F9FA")

    for i, model in enumerate(MODELS):
        values = [data[model][sk] for sk in SCORE_KEYS]
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
                fontsize=6.5, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=10)
    ax.set_ylim(0, 5.3)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.text(0.01, 0.97, "↑ higher is better",
            transform=ax.transAxes, fontsize=8, color="#888888", va="top")

axes[0].set_ylabel("Mean Score", fontsize=12)
axes[2].legend(fontsize=9, frameon=True, framealpha=0.9, loc="upper right")

fig.suptitle("Judge Method Comparison – Mean Score by Model & Category",
             fontsize=15, fontweight="bold", y=1.02)

plt.tight_layout()
out = Path(__file__).parent / "mad_comparison_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"저장 완료: {out}")
