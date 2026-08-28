#!/usr/bin/env python3
"""Simple black-and-white deployment flow chart, for the write-up."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle

OUT = "/home/mdsaifahmad/Franka/src/franka_openpi/docs/deployment_flow.png"

INK, GREY, BAND = "#000000", "#4a4a4a", "#f0f0f0"
DASH2 = (0, (4, 3))
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})

FW, FH = 13.4, 7.2
Y0, Y1 = 0.0, 100.0
fig = plt.figure(figsize=(FW, FH), dpi=200)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(Y0, Y1); ax.axis("off")
ASPECT = (FW / 100) / (FH / (Y1 - Y0))    # y-units per x-unit for a true circle

ax.text(3, 95.0, "Deployment — $\\pi$0.5 policy on a Franka",
        fontsize=19.5, fontweight="bold", va="center")

# ── lanes ────────────────────────────────────────────────────────────────────
GX, BX0, BX1 = 12.5, 14.5, 97.0        # label gutter, band extent
LANES = [("GPU\nSERVER", 69.0), ("LAPTOP", 45.0), ("FRANKA\nARM", 19.0)]
LH = 21.0
for name, yc in LANES:
    ax.add_patch(FancyBboxPatch((BX0, yc - LH / 2), BX1 - BX0, LH,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                fc=BAND, ec="none", zorder=0))
    ax.text(GX, yc, name, fontsize=10, fontweight="bold", color=GREY,
            va="center", ha="right", linespacing=1.5)

# ── boxes: (n, lane, title, subtitle) ────────────────────────────────────────
BOXES = [
    (1, 2, "Sense",    "3 cameras +\njoint angles"),
    (2, 1, "Pack",     "images, state,\nprompt"),
    (3, 0, "Think",    "$\\pi$0.5 predicts\n50 actions"),
    (4, 1, "Clean up", "smooth it,\nfade the seam"),
    (5, 1, "Send",     "publish as a\ntrajectory"),
    (6, 2, "Move",     "controller runs it\nat 1 kHz"),
]
BW, BH, GAP, X0 = 12.0, 15.0, 2.0, 15.5
geom = {}
for n, lane, title, sub in BOXES:
    x = X0 + (n - 1) * (BW + GAP)
    yc = LANES[lane][1]
    y = yc - BH / 2
    geom[n] = (x, y, yc)
    ax.add_patch(FancyBboxPatch((x, y), BW, BH,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                fc="white", ec=INK, lw=1.8, zorder=3))
    ax.add_patch(Ellipse((x, y + BH), 3.0, 3.0 * ASPECT,
                         fc=INK, ec="white", lw=1.4, zorder=7))
    ax.text(x, y + BH - 0.1, str(n), fontsize=8.2, color="white",
            fontweight="bold", ha="center", va="center", zorder=8)
    ax.text(x + BW / 2, y + BH - 5.0, title, fontsize=12.5, fontweight="bold",
            ha="center", va="center", zorder=5)
    ax.text(x + BW / 2, y + 4.8, sub, fontsize=9.2, color=GREY, ha="center",
            va="center", linespacing=1.6, zorder=5)

AR = "-|>,head_width=3.2,head_length=6"

def link(a, b, label=None):
    """Arrow from right edge of a into left edge of b, elbowed when lanes differ."""
    xa, _, ca = geom[a]; xb, _, cb = geom[b]
    x_out, x_in = xa + BW, xb
    xm = (x_out + x_in) / 2
    if abs(ca - cb) < 1e-6:
        ax.add_patch(FancyArrowPatch((x_out + 0.2, ca), (x_in - 0.2, ca),
                                     arrowstyle=AR, color=INK, lw=1.8,
                                     mutation_scale=1, zorder=4))
        lx, ly, ha = xm, ca + 1.6, "center"
    else:
        ax.plot([x_out + 0.2, xm], [ca, ca], color=INK, lw=1.8, zorder=4,
                solid_capstyle="round")
        ax.plot([xm, xm], [ca, cb], color=INK, lw=1.8, zorder=4,
                solid_capstyle="round")
        ax.add_patch(FancyArrowPatch((xm, cb), (x_in - 0.2, cb), arrowstyle=AR,
                                     color=INK, lw=1.8, mutation_scale=1, zorder=4))
        lx, ly, ha = xm + 1.0, (ca + cb) / 2, "left"   # beside the riser
    if label:
        ax.text(lx, ly, label, fontsize=8.8, color=GREY, ha=ha, va="bottom", zorder=6)

link(1, 2)
link(2, 3, "over the network")
link(3, 4, "0.3 – 0.5 s later")
link(4, 5)
link(5, 6)
# the one hop with no round trip: a topic publish preempts whatever is executing
ax.text(86.0, 32.6, "replaces the plan\nalready running", fontsize=8.8, color=GREY,
        ha="left", va="center", linespacing=1.7, zorder=6)

# ── feedback loop ────────────────────────────────────────────────────────────
x6, _, c6 = geom[6]
x1, _, c1 = geom[1]
Y, DASH = 6.5, (0, (5, 3))
ax.plot([x6 + BW / 2, x6 + BW / 2], [c6 - BH / 2, Y], color=INK, lw=1.8, ls=DASH,
        zorder=2, solid_capstyle="round")
ax.plot([x6 + BW / 2, x1 + BW / 2], [Y, Y], color=INK, lw=1.8, ls=DASH,
        zorder=2, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((x1 + BW / 2, Y), (x1 + BW / 2, c1 - BH / 2 - 0.2),
                             arrowstyle=AR, color=INK, lw=1.8, mutation_scale=1,
                             zorder=2, linestyle=DASH))
ax.text((x1 + x6) / 2 + BW / 2, Y + 1.5,
        "the next cycle starts while the arm is still moving — no pause at the seam",
        fontsize=10.5, color=INK, ha="center", va="bottom", fontweight="bold", zorder=6)

# ── nuance 1: RTC conditioning, routed over the top ──────────────────────────
x3, _, c3 = geom[3]
x5, _, c5 = geom[5]
TOP = 84.0
ax.plot([x5 + BW / 2, x5 + BW / 2], [c5 + BH / 2, TOP], color=INK, lw=1.5, ls=DASH2,
        zorder=2, solid_capstyle="round")
ax.plot([x5 + BW / 2, x3 + BW / 2], [TOP, TOP], color=INK, lw=1.5, ls=DASH2,
        zorder=2, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((x3 + BW / 2, TOP), (x3 + BW / 2, c3 + BH / 2 + 0.2),
                             arrowstyle=AR, color=INK, lw=1.5, mutation_scale=1,
                             zorder=2, linestyle=DASH2))
ax.text((x3 + x5) / 2 + BW / 2, TOP + 1.6,
        "every request carries the plan already running",
        fontsize=9.6, color=INK, ha="center", va="bottom", zorder=6)

# ── nuance 2: what "clean up" means ──────────────────────────────────────────
x4, y4, c4 = geom[4]
ax.plot([x4 + BW / 2, x4 + BW / 2], [c4 - BH / 2, 32.8], color=GREY, lw=1.0,
        zorder=2, solid_capstyle="round")
ax.text(56, 31.6, "the raw plan zigzags — filter it, then fade out the step "
                  "where old and new plans meet",
        fontsize=9.2, color=GREY, ha="center", va="center", zorder=6)

# ── the essence of RTC ──────────────────────────────────────────────────────
# Schematic, not to scale: the request fires at chunk index s (exec_horizon, 1 here,
# so in practice as soon as a chunk is published) and the black block is the d = 5
# actions that play while the reply is in flight. Those are sent as prev_actions and
# the server pins the reply's first d actions to them (Black, Galliker & Levine 2025,
# eq. 5 — beyond d the hold decays over the rest of the overlap rather than stopping
# dead, which is a refinement this drawing deliberately leaves out).
XA, L, TH = 22.0, 15.0, 2.3
yA, yB = 71.5, 66.0
FIRE, BLK = 0.22, 0.14           # where the request goes out, how much runs meanwhile
xq = XA + FIRE * L
wb = BLK * L

ax.text(15.0, 78.2, "what RTC does", fontsize=9.6, fontweight="bold", va="center",
        zorder=6)
ax.text(15.0, 75.8, "(real-time chunking: Black, Galliker & Levine, 2025)",
        fontsize=7.4, color=GREY, va="center", zorder=6)

# the chunk being executed, with the actions that will run during the wait in black
ax.text(21.2, yA, "now running", fontsize=7.4, color=GREY, ha="right", va="center",
        zorder=6)
ax.add_patch(Rectangle((XA, yA - TH / 2), L, TH, fc="white", ec=INK, lw=1.3, zorder=5))
ax.add_patch(Rectangle((xq, yA - TH / 2), wb, TH, fc=INK, ec=INK, lw=1.3, zorder=6))

# the reply: same block at its head, then its own continuation
ax.text(21.2, yB, "what comes back", fontsize=7.4, color=GREY, ha="right", va="center",
        zorder=6)
ax.add_patch(Rectangle((xq, yB - TH / 2), L, TH, fc="white", ec=INK, lw=1.3, zorder=5))
ax.add_patch(Rectangle((xq, yB - TH / 2), wb, TH, fc=INK, ec=INK, lw=1.3, zorder=6))

ax.plot([xq, xq], [yA + 1.7, yB - 1.5], color=INK, lw=1.0, ls=(0, (2.5, 2)), zorder=7)
ax.text(xq + 0.7, (yA + yB) / 2 - 0.1, "the request goes out here", fontsize=7.6,
        color=INK, ha="left", va="center", zorder=7)

ax.text(xq + wb / 2, yA + 1.9, "0.3 – 0.5 s", fontsize=7.4, color=GREY,
        ha="center", va="center", zorder=6)
ax.text(xq, yB - 2.4,
        "the arm moves through the black,\nso the reply starts from there",
        fontsize=7.8, color=GREY, ha="left", va="top", linespacing=1.7, zorder=6)

fig.savefig(OUT, dpi=200)
print("wrote", OUT)
