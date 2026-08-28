#!/usr/bin/env python3
"""Two-panel plot: what open-loop does with a chunk vs what RTC does.

All curves are real logged data from debug/rtc/rtc_sim_commanded.npz.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/mdsaifahmad/Franka/src/franka_openpi"
OUT = f"{REPO}/docs/rtc_explained.png"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
RED, INK, INK2, MUTED = "#d03b3b", "#0b0b0b", "#52514e", "#8a8985"
SURF, LINE = "#fcfcfb", "#dedcd7"
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF,
                     "savefig.facecolor": SURF, "axes.facecolor": "white",
                     "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": LINE})

z = np.load(f"{REPO}/debug/rtc/rtc_sim_commanded.npz")
N = int(z["n_publishes"])
t_pub = np.array([float(z[f"t_{k}"]) for k in range(N)]); t_pub -= t_pub[0]
offs = [int(z[f"offset_{k}"]) for k in range(N)]
sj = np.load(f"{REPO}/debug/rtc/rtc_sim_splice_jumps.npy")
sm = np.load(f"{REPO}/debug/rtc/rtc_sim_splice_meta.npy")
rtt = np.load(f"{REPO}/debug/rtc/rtc_sim_rtt.npy")
RTT50 = float(np.percentile(rtt, 50))
jump_mx = np.abs(sj).max(axis=1)
J50, J95 = np.percentile(jump_mx, 50), np.percentile(jump_mx, 95)

J = 5                      # fer_joint6 — the joint that moves most in this episode
DT, D_SENT, BLEND = 0.1, 5, 6
XMAX = 7.0

# what the arm was actually commanded, chunk by chunk, as RTC replaced them
T, Y = [], []
for k in range(N):
    st = z[f"step_times_{k}"]; a = z[f"actions_{k}"]
    t_end = t_pub[k + 1] if k + 1 < N else t_pub[k] + st[-1]
    m = (t_pub[k] + st) < t_end + 1e-9
    T.append(t_pub[k] + st[m]); Y.append(a[m, J])
T = np.concatenate(T); Y = np.concatenate(Y)

# the very first chunk, played open-loop to its end
a0, st0 = z["actions_0"], z["step_times_0"]
T0, Y0 = st0, a0[:, J]

fig = plt.figure(figsize=(13.2, 9.2), dpi=200)
fig.text(0.035, 0.965, "One chunk of 50 actions — and three things you can do with it",
         fontsize=19, fontweight="bold", va="center")
fig.text(0.035, 0.932, "Franka FER · $\\pi$0.5 · joint 6 commanded target · every curve is logged "
                       "data from the same 18 s episode",
         fontsize=10.4, color=INK2, va="center")

# ══ TOP ═════════════════════════════════════════════════════════════════════
mvis = T <= XMAX
ax1 = fig.add_axes([0.062, 0.545, 0.905, 0.315])
ax1.fill_between(T0, Y0, np.interp(T0, T, Y), color=VIOLET, alpha=0.09, lw=0, zorder=2)
ax1.plot(T0, Y0, "-", color=ORANGE, lw=2.6, zorder=4,
         label="open-loop: play all 50 actions of the first chunk (5.5 s)")
ax1.plot(T0, Y0, ".", color=ORANGE, ms=4, zorder=4)
t_stop0, t_stop1 = T0[-1], T0[-1] + RTT50
ax1.plot([t_stop0, t_stop1], [Y0[-1], Y0[-1]], "-", color=RED, lw=3.4, zorder=5)
ax1.axvspan(t_stop0, t_stop1, color=RED, alpha=0.12, lw=0, zorder=1)
ax1.plot(T[mvis], Y[mvis], "-", color=AQUA, lw=2.6, zorder=6,
         label="RTC: the same policy, replanned every 0.35 s")

vis = [k for k in range(1, N) if t_pub[k] < XMAX]
ax1.plot(t_pub[vis], np.interp(t_pub[vis], T, Y), "v", color=AQUA, ms=6.5,
         mec="white", mew=0.8, zorder=7)

gap = np.interp(t_stop0, T, Y) - Y0[-1]
ax1.annotate("", xy=(t_stop0, Y0[-1]), xytext=(t_stop0, Y0[-1] + gap),
             arrowprops=dict(arrowstyle="<->", color=VIOLET, lw=2.0), zorder=8)

ax1.set_ylim(1.58, 2.54)
ax1.text((t_stop0 + t_stop1) / 2, 1.605, f"arm stopped\n{RTT50*1e3:.0f} ms\nevery chunk",
         fontsize=8.8, color=RED, ha="center", va="bottom", fontweight="bold", linespacing=1.5)
ax1.text(t_stop0 - 0.18, 1.70,
         f"{abs(gap):.2f} rad  ({abs(gap)*180/np.pi:.0f}°) apart\nby the time the open-loop\nchunk runs out",
         fontsize=8.8, color=VIOLET, ha="right", va="center", fontweight="bold", linespacing=1.5)
ax1.text(0.16, 2.29,
         f"▼  a new chunk spliced in — {len(vis)} of them before the open-loop chunk would have finished",
         fontsize=9.0, color="#0f7f59", ha="left", va="top", fontweight="bold")

ax1.set_xlim(0, XMAX); ax1.set_xlabel("time (s)", fontsize=10)
ax1.set_ylabel("joint 6 target (rad)", fontsize=10)
ax1.legend(fontsize=9.4, frameon=False, loc="upper left", handlelength=2.4)
ax1.grid(color=LINE, lw=0.8); ax1.set_axisbelow(True)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)
ax1.tick_params(labelsize=9)
ax1.set_title("A chunk is a 5-second prediction. Executing it open-loop means acting on stale "
              "information — and stopping at the end of every one.",
              fontsize=11, color=INK, loc="left", pad=9, fontweight="bold")

# ══ BOTTOM ══════════════════════════════════════════════════════════════════
ax2 = fig.add_axes([0.062, 0.085, 0.905, 0.315])
kk = 12                                        # a splice whose jump sits at the run median
a_old, a_new = z[f"actions_{kk}"], z[f"actions_{kk + 1}"]
ro = int(sm[kk, 3]) - offs[kk]
nb = min(BLEND, len(a_new))
ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(nb) / nb))
raw_new = np.array(a_new[:, :7], dtype=float)
raw_new[:nb] -= ramp[:, None] * (-sj[kk])[None, :]

base = a_old[ro, J]
NF = 16
x_old = np.arange(-ro, 1)
x_new = np.arange(0, NF + 1)
y_old = (a_old[ro - ro:ro + 1, J] - base) * 1e3
y_raw = (raw_new[:NF + 1, J] - base) * 1e3
y_bld = (a_new[:NF + 1, J] - base) * 1e3

ax2.axvspan(0, D_SENT, color=VIOLET, alpha=0.10, lw=0)
ax2.plot(x_old, y_old, "-o", color=BLUE, lw=2.6, ms=6,
         label="what the arm was already committed to (previous chunk)")
ax2.plot(x_new, y_raw, "--x", color=ORANGE, lw=2.2, ms=7,
         label="RTC conditioning only — the reply spliced in verbatim")
ax2.plot(x_new, y_bld, "-o", color=AQUA, lw=2.6, ms=6,
         label="RTC + raised-cosine blend — what the arm was actually sent")
ax2.annotate("", xy=(0, y_raw[0]), xytext=(0, 0),
             arrowprops=dict(arrowstyle="<->", color=RED, lw=2.0))
ax2.text(-0.25, y_raw[0] / 2 + 12, f"{abs(sj[kk, J])*1e3:.0f} mrad step\nin one control cycle",
         fontsize=9.0, color=RED, ha="right", va="center", fontweight="bold", linespacing=1.5)
ytop = ax2.get_ylim()[1]
ax2.set_ylim(ax2.get_ylim()[0], ytop * 1.16)
ytop = ax2.get_ylim()[1]
ax2.text(D_SENT / 2, ytop * 0.88, "hard-frozen · d = 5 actions\npinned to the previous chunk",
         fontsize=8.8, color=VIOLET, ha="center", va="top", fontweight="bold", linespacing=1.5)
yb_ = ytop * 0.955
ax2.annotate("", xy=(0, yb_), xytext=(BLEND, yb_),
             arrowprops=dict(arrowstyle="<->", color=AQUA, lw=1.8))
ax2.text(BLEND + 0.25, yb_, "raised-cosine blend, 6 steps", fontsize=8.8, color="#0f7f59",
         ha="left", va="center", fontweight="bold")
ax2.text(NF, ytop * 0.30, "with no conditioning at all the two chunks are\n"
                          "independent samples — probed at 83 mrad apart",
         fontsize=8.8, color=MUTED, ha="right", va="top", linespacing=1.5)

ax2.set_xlim(-ro - 0.4, NF + 0.4)
ax2.set_xlabel("policy step, relative to the splice   (1 step = 0.1 s)", fontsize=10)
ax2.set_ylabel("joint 6 target (mrad, relative)", fontsize=10)
ax2.legend(fontsize=9.4, frameon=False, loc="lower right", handlelength=2.4)
ax2.grid(color=LINE, lw=0.8); ax2.set_axisbelow(True)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
ax2.tick_params(labelsize=9)
ax2.set_title("Zoom on one handover. Replanning early is not enough — the new chunk has to agree "
              "with the motion already committed.",
              fontsize=11, color=INK, loc="left", pad=9, fontweight="bold")

fig.text(0.062, 0.021,
         f"Measured over the episode: {len(sj)} splices, raw residual p50 {J50*1e3:.0f} mrad / "
         f"p95 {J95*1e3:.0f} mrad — after blending, the commanded step at every splice is exactly 0.",
         fontsize=9.2, color=MUTED, va="center")

fig.savefig(OUT, dpi=200)
print("wrote", OUT)
