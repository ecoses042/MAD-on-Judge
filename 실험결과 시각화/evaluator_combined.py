import matplotlib.pyplot as plt
import numpy as np

# ── 데이터 ────────────────────────────────────────────────────────────────────
data = {
    "GPT4o-mini": {
        "Evaluator 1": {"rmse": 0.956, "rho": 0.332},
        "Evaluator 2": {"rmse": 0.906, "rho": 0.285},
        "Average":     {"rmse": 0.847, "rho": 0.377},
    },
    "Qwen3.5-9B": {
        "Evaluator 1": {"rmse": 0.894, "rho": 0.307},
        "Evaluator 2": {"rmse": 0.919, "rho": 0.208},
        "Average":     {"rmse": 0.821, "rho": 0.311},
    },
    "Gemma3-4B": {
        "Evaluator 1": {"rmse": 0.809, "rho": 0.277},
        "Evaluator 2": {"rmse": 0.836, "rho": 0.135},
        "Average":     {"rmse": 0.717, "rho": 0.250},
    },
    "Llama3-8B": {
        "Evaluator 1": {"rmse": 1.101, "rho": -0.001},
        "Evaluator 2": {"rmse": 1.012, "rho":  0.056},
        "Average":     {"rmse": 0.986, "rho":  0.029},
    },
}

models     = list(data.keys())
evaluators = ["Evaluator 1", "Evaluator 2", "Average"]

MODEL_COLORS = {
    "GPT4o-mini": "#DD8452",
    "Qwen3.5-9B": "#55A868",
    "Gemma3-4B":  "#4C72B0",
    "Llama3-8B":  "#C44E52",
}

# x축 = Evaluator 그룹, 각 그룹 안에 모델별 막대
x         = np.arange(len(evaluators))
bar_width = 0.18
offsets   = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * bar_width

# ── 공통 그리기 함수 ──────────────────────────────────────────────────────────
def draw_bars(ax, metric, ylabel, title, ylim, value_offset=0.012, fmt="{:.3f}"):
    ax.set_facecolor("#F8F9FA")
    for i, model in enumerate(models):
        values = [data[model][ev][metric] for ev in evaluators]
        bars = ax.bar(x + offsets[i], values,
                      width=bar_width,
                      color=MODEL_COLORS[model],
                      alpha=0.85,
                      label=model,
                      zorder=3)
        for bar, v in zip(bars, values):
            va   = "bottom" if v >= 0 else "top"
            ypos = (bar.get_height() if v >= 0 else 0) + (value_offset if v >= 0 else -value_offset)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    ypos,
                    fmt.format(v),
                    ha="center", va=va,
                    fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(evaluators, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=9.5, frameon=True, framealpha=0.9, loc="upper right")

# ── 서브플롯 ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#F8F9FA")

draw_bars(axes[0],
          metric="rmse",
          ylabel="RMSE",
          title="RMSE by Model & Evaluator",
          ylim=(0.0, 1.45),
          value_offset=0.015)

draw_bars(axes[1],
          metric="rho",
          ylabel="Spearman ρ",
          title="Spearman ρ by Model & Evaluator",
          ylim=(-0.05, 0.55),
          value_offset=0.008)

axes[0].text(0.01, 0.97, "↓ lower is better",
             transform=axes[0].transAxes,
             fontsize=9, color="#888888", va="top")
axes[1].text(0.01, 0.97, "↑ higher is better",
             transform=axes[1].transAxes,
             fontsize=9, color="#888888", va="top")

fig.suptitle("Evaluator Comparison - Content",
             fontsize=16, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/evaluator_combined.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("저장 완료: evaluator_combined.png")