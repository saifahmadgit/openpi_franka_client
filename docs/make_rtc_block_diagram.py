#!/usr/bin/env python3
"""Single block diagram: the RTC + smoothing pipeline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "/home/mdsaifahmad/Franka/src/franka_openpi/docs/rtc_block_diagram.png"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURF = "#fcfcfb"

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF,
                     "savefig.facecolor": SURF})

fig = plt.figure(figsize=(15.5, 6.6), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

ax.text(2, 94, "RTC deployment pipeline — Franka FER + $\\pi$0.5", fontsize=19,
        fontweight="bold", va="center")
ax.text(2, 86.5, "The next inference fires at chunk index s = 1 while the arm is still moving, "
                 "so the arm never stops at a chunk boundary.",
        fontsize=10.5, color=INK2, va="center")

BLOCKS = [
    ("Franka FER",         ["3× RealSense 640×480", "/joint_states @ 1.4 kHz"], ORANGE),
    ("openpi_rtc_node",    ["pack observation:", "images, state(9), prompt"],   BLUE),
    ("$\\pi$0.5 policy server", ["GPU · websocket", "→ 50-action chunk"],       VIOLET),
    ("Savitzky–Golay",     ["w = 15, p = 4", "removes chunk zigzag"],           AQUA),
    ("splice blend",       ["raised cosine, 6 steps", "removes the handover step"], AQUA),
    ("rtc_executor",       ["time-parameterize,", "publish to the JTC topic"],  BLUE),
    ("fer_arm_controller", ["1 kHz · replaces the", "running trajectory"],      ORANGE),
]

BW, GAP, X0 = 12.0, 2.0, 2.0
YB, BH = 50, 26
cx = []
for i, (title, subs, c) in enumerate(BLOCKS):
    x = X0 + i * (BW + GAP)
    cx.append(x + BW / 2)
    ax.add_patch(FancyBboxPatch((x, YB), BW, BH, boxstyle="round,pad=0,rounding_size=1.0",
                                fc="white", ec=c, lw=2.0, zorder=3))
    ax.add_patch(FancyBboxPatch((x, YB + BH - 8.5), BW, 8.5,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                fc=c, ec="none", alpha=0.14, zorder=2))
    ax.text(x + BW / 2, YB + BH - 4.4, title, fontsize=10.4, fontweight="bold", color=c,
            ha="center", va="center", zorder=5)
    ax.text(x + BW / 2, YB + 8.5, "\n".join(subs), fontsize=8.2, color=INK2, ha="center",
            va="center", linespacing=1.8, zorder=5)

for i in range(len(BLOCKS) - 1):
    x0 = X0 + i * (BW + GAP) + BW
    ax.add_patch(FancyArrowPatch((x0 + 0.15, YB + BH / 2), (x0 + GAP - 0.15, YB + BH / 2),
                                 arrowstyle="-|>,head_width=3.6,head_length=6",
                                 color=MUTED, lw=2.0, mutation_scale=1, zorder=6))

def elbow(x_from, x_to, y, color, label, lw=2.0):
    ax.plot([x_from, x_from], [YB, y], color=color, lw=lw, zorder=2,
            solid_capstyle="round")
    ax.plot([x_from, x_to], [y, y], color=color, lw=lw, zorder=2, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((x_to, y), (x_to, YB - 0.4),
                                 arrowstyle="-|>,head_width=3.6,head_length=6",
                                 color=color, lw=lw, mutation_scale=1, zorder=2))
    ax.text((x_from + x_to) / 2, y + 2.0, label, fontsize=8.8, color=color, ha="center",
            va="bottom", fontweight="bold", linespacing=1.6)

elbow(cx[4], cx[2], 32,
      VIOLET, "prev_actions (50,14) + s + d  —  the RTC conditioning:\n"
              "the server pins the actions already committed to the arm")
elbow(cx[6], cx[1], 16, ORANGE, "measured joint position and velocity")

ax.text(2.0, 5.0, "step_duration 0.1 s   ·   H = 50   ·   exec_horizon s = 1   ·   rtc_d = 5   ·   "
                  "smooth_window 15 / polyorder 4   ·   splice_blend_steps 6",
        fontsize=9.0, color=MUTED, va="center", family="monospace")

fig.savefig(OUT, dpi=200)
print("wrote", OUT)
