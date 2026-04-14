import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# ── 한국어 폰트 설정 (Windows: Malgun Gothic) ────────────────────────────────
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 데이터 ────────────────────────────────────────────────────────────────────
N = [
    10_000, 20_000, 30_000, 40_000, 50_000,
    60_000, 70_000, 80_000, 90_000, 100_000,
    110_000, 120_000, 130_000, 140_000, 150_000,
    160_000, 170_000, 180_000, 190_000, 200_000,
    210_000, 220_000, 230_000, 240_000, 250_000,
    260_000, 270_000, 280_000, 290_000, 300_000,
    310_000, 320_000, 330_000, 340_000, 350_000,
    360_000, 370_000, 380_000, 390_000, 400_000,
    410_000, 420_000, 430_000, 440_000, 450_000,
    460_000, 470_000, 480_000, 490_000, 500_000,
]

cpu_ms = [
    0.011, 0.021, 0.031, 0.050, 0.052,
    0.067, 0.102, 0.128, 0.134, 0.167,
    0.236, 0.189, 0.133, 0.154, 0.177,
    0.201, 0.252, 0.185, 0.272, 0.231,
    0.253, 0.260, 0.249, 0.278, 0.276,
    0.297, 0.294, 0.330, 0.312, 0.407,
    0.352, 0.373, 0.360, 0.367, 0.396,
    0.468, 0.426, 0.421, 0.437, 0.442,
    0.436, 0.455, 0.506, 0.470, 0.542,
    0.495, 0.519, 0.576, 0.524, 0.582,
]

gpu_ms = [
    23.492, 36.390, 36.113, 33.774, 22.716,
    26.693, 21.815, 29.051, 21.954, 22.456,
    22.908, 41.433, 25.010, 22.412, 24.661,
    24.088, 21.886, 22.224, 22.078, 22.595,
    22.206, 25.092, 22.586, 22.362, 22.824,
    23.808, 23.704, 23.163, 25.129, 26.137,
    22.377, 17.944, 22.518, 21.384, 20.488,
    23.643, 20.784, 20.397, 20.663, 18.894,
    23.915, 20.733, 20.853, 19.936, 20.657,
    19.170, 20.507, 20.208, 18.828, 1.738,   # 500k: 이상치(*)
]

N_arr   = np.array(N)
cpu_arr = np.array(cpu_ms)
gpu_arr = np.array(gpu_ms)

# 500k GPU 이상치 분리
ANOMALY_IDX = len(N) - 1   # 마지막 인덱스

# ── 추세선 (선형 회귀) ───────────────────────────────────────────────────────
cpu_coef = np.polyfit(N_arr, cpu_arr, 1)
gpu_coef = np.polyfit(N_arr[:-1], gpu_arr[:-1], 1)   # 이상치 제외

x_fit   = np.linspace(N_arr[0], N_arr[-1], 500)
cpu_fit = np.polyval(cpu_coef, x_fit)
gpu_fit = np.polyval(gpu_coef, x_fit)

# ── 색상 ─────────────────────────────────────────────────────────────────────
CPU_COLOR     = "#4C72B0"
GPU_COLOR     = "#C44E52"
TREND_ALPHA   = 0.75
BG_COLOR      = "#F8F9FA"
ANOMALY_COLOR = "#FF8C00"

# ── 플롯 ─────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG_COLOR)
ax1.set_facecolor(BG_COLOR)

ax2 = ax1.twinx()
ax2.set_facecolor(BG_COLOR)

# ── CPU (왼쪽 y축) ───────────────────────────────────────────────────────────
ax1.plot(N_arr, cpu_arr,
         color=CPU_COLOR, linewidth=1.4, alpha=0.55,
         marker="o", markersize=4, label="CPU 실측")
ax1.plot(x_fit, cpu_fit,
         color=CPU_COLOR, linewidth=2.2, linestyle="--",
         alpha=TREND_ALPHA, label="CPU 추세선")

# ── GPU (오른쪽 y축) — 이상치 제외 ──────────────────────────────────────────
ax2.plot(N_arr[:-1], gpu_arr[:-1],
         color=GPU_COLOR, linewidth=1.4, alpha=0.55,
         marker="s", markersize=4, label="GPU 실측")
ax2.plot(x_fit, gpu_fit,
         color=GPU_COLOR, linewidth=2.2, linestyle="--",
         alpha=TREND_ALPHA, label="GPU 추세선")

# ── 이상치 마킹 ──────────────────────────────────────────────────────────────
ax2.scatter([N_arr[ANOMALY_IDX]], [gpu_arr[ANOMALY_IDX]],
            color=ANOMALY_COLOR, zorder=5, s=80, marker="*",
            label=f"GPU 이상치 (N=500k, {gpu_arr[ANOMALY_IDX]:.3f} ms*)")
ax2.annotate(f"{gpu_arr[ANOMALY_IDX]:.3f} ms*",
             xy=(N_arr[ANOMALY_IDX], gpu_arr[ANOMALY_IDX]),
             xytext=(-55, 12), textcoords="offset points",
             fontsize=8.5, color=ANOMALY_COLOR,
             arrowprops=dict(arrowstyle="->", color=ANOMALY_COLOR, lw=1.2))

# ── 축 레이블 ─────────────────────────────────────────────────────────────────
ax1.set_xlabel("N (데이터 크기)", fontsize=12)
ax1.set_ylabel("CPU 시간 (ms)", fontsize=12, color=CPU_COLOR)
ax2.set_ylabel("GPU 시간 (ms)", fontsize=12, color=GPU_COLOR)

ax1.tick_params(axis="y", labelcolor=CPU_COLOR)
ax2.tick_params(axis="y", labelcolor=GPU_COLOR)

ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
ax1.tick_params(axis="x", rotation=30)

# ── 범례 통합 ─────────────────────────────────────────────────────────────────
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           fontsize=9, frameon=True, framealpha=0.92,
           loc="upper left")

# ── 그리드 & 스파인 ───────────────────────────────────────────────────────────
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)
for spine in ["top"]:
    ax1.spines[spine].set_visible(False)
    ax2.spines[spine].set_visible(False)

# ── 추세선 기울기 주석 ────────────────────────────────────────────────────────
cpu_slope_per_10k = cpu_coef[0] * 10_000
gpu_slope_per_10k = gpu_coef[0] * 10_000
ax1.text(0.01, 0.97,
         f"CPU 기울기: +{cpu_slope_per_10k:.4f} ms / 10k",
         transform=ax1.transAxes, fontsize=8.5,
         color=CPU_COLOR, va="top", alpha=0.85)
ax1.text(0.01, 0.91,
         f"GPU 기울기: {gpu_slope_per_10k:+.4f} ms / 10k  (이상치 제외)",
         transform=ax1.transAxes, fontsize=8.5,
         color=GPU_COLOR, va="top", alpha=0.85)

# ── 제목 & 저장 ───────────────────────────────────────────────────────────────
fig.suptitle("CPU vs GPU 연산 시간 추세 (N 크기별)",
             fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
out = Path(__file__).parent / "cpu_gpu_time_trend.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.show()
print(f"저장 완료: {out}")
