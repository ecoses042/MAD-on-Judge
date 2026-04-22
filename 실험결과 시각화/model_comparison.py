import matplotlib.pyplot as plt
import numpy as np

# ── 데이터 ────────────────────────────────────────────────────────────────────
data = {
    "GPT4o-mini":  {"content": {"rmse": 0.844, "rho": 0.377},
                    "organization": {"rmse": 0.868, "rho": 0.377},
                    "expression": {"rmse": 0.554, "rho": 0.516}},
    "Qwen3.5-9B":  {"content": {"rmse": 0.906, "rho": 0.311},
                    "organization": {"rmse": 0.827, "rho": 0.508},
                    "expression": {"rmse": 0.669, "rho": 0.423}},
    "Gemma3-4B":   {"content": {"rmse": 0.940, "rho": 0.250},
                    "organization": {"rmse": 0.969, "rho": 0.336},
                    "expression": {"rmse": 0.708, "rho": 0.343}},
    "Llama3-8B":   {"content": {"rmse": 1.173, "rho": 0.029},
                    "organization": {"rmse": 1.164, "rho": 0.092},
                    "expression": {"rmse": 0.908, "rho": 0.120}},
}

models     = list(data.keys())
categories = ["content", "organization", "expression"]
CAT_LABELS = ["Content", "Organization", "Expression"]

MODEL_COLORS = {
    "GPT4o-mini": "#DD8452",
    "Qwen3.5-9B": "#55A868",
    "Gemma3-4B":  "#4C72B0",
    "Llama3-8B":  "#C44E52",
}

x         = np.arange(len(categories))
bar_width = 0.18
offsets   = np.linspace(-(len(models)-1)/2, (len(models)-1)/2, len(models)) * bar_width

# ── 공통 그리기 함수 ──────────────────────────────────────────────────────────
def draw_bars(ax, metric, ylabel, title, ylim, value_offset=0.012, fmt="{:.3f}"):
    ax.set_facecolor("#F8F9FA")
    for i, model in enumerate(models):
        values = [data[model][cat][metric] for cat in categories]
        bars = ax.bar(x + offsets[i], values,
                      width=bar_width,
                      color=MODEL_COLORS[model],
                      alpha=0.85,
                      label=model,
                      zorder=3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + value_offset,
                    fmt.format(v),
                    ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(CAT_LABELS, fontsize=12)
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
          title="RMSE by Model & Category",
          ylim=(0.0, 1.45),
          value_offset=0.015)

draw_bars(axes[1],
          metric="rho",
          ylabel="Spearman ρ",
          title="Spearman ρ by Model & Category",
          ylim=(0.0, 0.70),
          value_offset=0.008)

axes[0].text(0.01, 0.97, "↓ lower is better",
             transform=axes[0].transAxes,
             fontsize=9, color="#888888", va="top")
axes[1].text(0.01, 0.97, "↑ higher is better",
             transform=axes[1].transAxes,
             fontsize=9, color="#888888", va="top")

fig.suptitle("Model Comparison",
             fontsize=16, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("judge_rmse_rho_bar.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("저장 완료: judge_rmse_rho_bar.png")