import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_json_with_fallback(*relative_paths):
    base = Path(__file__).parent.parent
    for rel in relative_paths:
        path = base / rel
        if path.exists():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"입력 JSON을 찾지 못했습니다: {relative_paths}")


# ── 데이터 로드 ───────────────────────────────────────────────────────────────
raw = load_json_with_fallback("stats/mad_a_base_stats_all.json", "stats/mad4_stats_all.json")

MODEL_LABELS = {
    "mad_a_base_results_gemma": "Gemma3-4B",
    "mad4_results_gemma": "Gemma3-4B",
    "mad_a_base_results_qwen": "Qwen3.5-9B",
    "mad4_results_qwen": "Qwen3.5-9B",
    "mad_a_base_results_llama": "Llama3-8B",
    "mad4_results_llama": "Llama3-8B",
    "mad_a_base_results_gpt": "GPT4o-mini",
    "mad4_results_gpt": "GPT4o-mini",
}
MODEL_COLORS = {
    "Gemma3-4B":  "#4C72B0",
    "Qwen3.5-9B": "#55A868",
    "Llama3-8B":  "#C44E52",
    "GPT4o-mini": "#DD8452",
}

SCORE_KEYS = ["domain_match", "score_rationale_consistency", "specificity", "groundedness"]
CAT_LABELS = ["Domain Match", "Score-Rationale\nConsistency", "Specificity", "Groundedness"]

data = {}
for key, label in MODEL_LABELS.items():
    if key in raw and label not in data:
        data[label] = raw[key]
models = list(data.keys())

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
fig.suptitle("MAD-A Base – Model Comparison",
             fontsize=16, fontweight="bold", y=1.02)

plt.tight_layout()
out = Path(__file__).parent / "mad4_mean_bar.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"저장 완료: {out}")
