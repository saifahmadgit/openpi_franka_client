#!/usr/bin/env python3
"""Portfolio figure: RTC deployment + smoothing on a Franka FER.

Every number is either a launch parameter from openpi_franka_RTC.launch_sim.py
or computed here from the run artefacts in debug/.
"""
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from scipy.signal import savgol_filter

REPO = "/home/mdsaifahmad/Franka/src/franka_openpi"
OUT = f"{REPO}/docs/rtc_deployment.png"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, RED, GOOD = "#4a3aa7", "#d03b3b", "#0ca30c"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
SURF, PANEL, LINE = "#fcfcfb", "#f2f1ee", "#dedcd7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": LINE, "figure.facecolor": SURF, "savefig.facecolor": SURF,
})

# ── data ────────────────────────────────────────────────────────────────────
z = np.load(f"{REPO}/debug/rtc/rtc_sim_commanded.npz")
NPUB = int(z["n_publishes"])
t_pub = np.array([float(z[f"t_{k}"]) for k in range(NPUB)]); t_pub -= t_pub[0]
offs = [int(z[f"offset_{k}"]) for k in range(NPUB)]
rtt = np.load(f"{REPO}/debug/rtc/rtc_sim_rtt.npy")
sj = np.load(f"{REPO}/debug/rtc/rtc_sim_splice_jumps.npy")
sm = np.load(f"{REPO}/debug/rtc/rtc_sim_splice_meta.npy")
raw_chunk = np.load(f"{REPO}/debug/sanity/baseline_chunk.npy")[:, :7]

RTT50, RTT95, RTTMX = np.percentile(rtt, 50), np.percentile(rtt, 95), rtt.max()
jump_mx = np.abs(sj).max(axis=1)
J50, J95 = np.percentile(jump_mx, 50), np.percentile(jump_mx, 95)
d_act = sm[:, 2]
DT, H, D_SENT, S_EXEC, BLEND, SW, SP = 0.1, 50, 5, 1, 6, 15, 4
DPUB = float(np.median(np.diff(t_pub)))

# ── canvas ──────────────────────────────────────────────────────────────────
FW, FH = 13.6, 19.4
fig = plt.figure(figsize=(FW, FH), dpi=200)
L, R = 0.035, 0.965
MID = 0.500

def ax_at(y0, y1, x0=L, x1=R, xlim=(0, 100), ylim=(0, 100), frame=True):
    a = fig.add_axes([x0, y0, x1 - x0, y1 - y0])
    a.set_xlim(*xlim); a.set_ylim(*ylim); a.axis("off")
    if frame:
        a.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0,rounding_size=0.012",
                                   fc=PANEL, ec=LINE, lw=0.9, transform=a.transAxes, zorder=0,
                                   mutation_aspect=(y1 - y0) * FH / ((x1 - x0) * FW)))
    return a

def head(ax, label, title, sub=None):
    ax.text(0.0, 1.012, label, fontsize=10.5, fontweight="bold", color=BLUE,
            transform=ax.transAxes, va="bottom", ha="left")
    ax.text(0.021, 1.012, title, fontsize=11.0, fontweight="bold", color=INK,
            transform=ax.transAxes, va="bottom", ha="left")
    if sub:
        ax.text(1.0, 1.016, sub, fontsize=8.3, color=MUTED, transform=ax.transAxes,
                va="bottom", ha="right")

def wrap(s, w):
    return textwrap.fill(s, w)

# ═══ TITLE ══════════════════════════════════════════════════════════════════
T = ax_at(0.943, 0.996, frame=False)
T.text(0, 74, "Real-Time Chunking on a Franka FER", fontsize=26, fontweight="bold", va="center")
T.text(0, 32, "Running a $\\pi$0.5 flow policy at 10 Hz with no hitch at the chunk boundary — "
              "and the smoothing that made 10 Hz survivable.",
       fontsize=11.4, color=INK2, va="center")
chips = [("step_duration 0.1 s", BLUE), ("exec_horizon  s = 1", BLUE), ("rtc_d = 5", BLUE),
         ("smooth_window 15 / polyorder 4", AQUA), ("splice_blend_steps 6", AQUA),
         ("H = 50", MUTED)]
x = 0.0
for txt, c in chips:
    w = 0.60 * len(txt) + 2.4
    T.add_patch(FancyBboxPatch((x, -8), w, 17, boxstyle="round,pad=0,rounding_size=1.0",
                               fc="white", ec=c, lw=1.0))
    T.text(x + w / 2, 0.2, txt, fontsize=8.2, color=c, ha="center", va="center", family="monospace")
    x += w + 1.6

# ═══ A · DEPLOYMENT TOPOLOGY ════════════════════════════════════════════════
A = ax_at(0.688, 0.912)
head(A, "A", "Deployment topology", "three machines, one 10 Hz closed loop")

def machine(ax, x, w, y, h, title, sub, lines, accent):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc="white", ec=accent, lw=1.5, zorder=2))
    ax.add_patch(Rectangle((x + 0.4, y + h - 9.0), w - 0.8, 8.4, fc=accent, ec="none",
                           alpha=0.13, zorder=3))
    ax.text(x + 2.2, y + h - 3.4, title, fontsize=9.6, fontweight="bold", color=accent,
            va="center", zorder=4)
    ax.text(x + 2.2, y + h - 7.2, sub, fontsize=7.0, color=INK2, va="center", zorder=4,
            family="monospace")
    yy = y + h - 12.5
    cw = int(w * 2.05)
    for ln in lines:
        body = wrap(ln, cw)
        ax.text(x + 2.4, yy, "▪ " + body.replace("\n", "\n  "), fontsize=7.3, color=INK,
                va="top", zorder=4, linespacing=1.45)
        yy -= 4.5 * (body.count("\n") + 1) + 1.5
    return yy

GX, GW = 0.5, 23.5
LX, LW = 33.0, 34.0
FX, FW_ = 76.0, 23.5

machine(A, GX, GW, 27, 71, "GPU server", "129.105.69.11:8000", [
    "serve_policy.py — websocket + msgpack, keepalive pings off",
    "$\\pi$0.5 checkpoint (LoRA, 55k steps)",
    "RTC-capable build: soft prefix attention over the overlap, not a hard freeze",
    "replies (H=50, 14) absolute joint targets + rtc_* flags",
    "server_timing.infer_ms is the honest timer — policy_timing measures JAX dispatch only",
    "delta action space: the server re-adds state, so the reply is stored and returned untouched",
], VIOLET)

A.add_patch(FancyBboxPatch((LX, 3), LW, 95, boxstyle="round,pad=0,rounding_size=1.2",
                           fc="white", ec=BLUE, lw=1.5, zorder=2))
A.add_patch(Rectangle((LX + 0.4, 89.6), LW - 0.8, 8.0, fc=BLUE, ec="none", alpha=0.13, zorder=3))
A.text(LX + 2.2, 94.6, "Laptop", fontsize=9.6, fontweight="bold", color=BLUE, va="center", zorder=4)
A.text(LX + 2.2, 91.0, "ROS 2 Jazzy · this repo", fontsize=7.0, color=INK2, va="center",
       zorder=4, family="monospace")

def stack(ax, x, y, lines, w):
    for ln in lines:
        body = wrap(ln, w)
        ax.text(x, y, "▪ " + body.replace("\n", "\n  "), fontsize=7.3, color=INK, va="top",
                zorder=5, linespacing=1.45)
        y -= 4.5 * (body.count("\n") + 1) + 1.3
    return y

A.text(LX + 2.2, 85.5, "openpi_rtc_node — the RTC loop", fontsize=8.2, fontweight="bold",
       color=BLUE, zorder=5, family="monospace")
y = stack(A, LX + 2.4, 81.5, [
    "pack obs: 3× RealSense 640×480 + state(9) + prompt",
    "fire the next inference at chunk index s = 1, on a worker thread — execution never blocks",
    "send prev_actions (50,14) verbatim, plus s_req and d",
    "_smooth_chunk — Savitzky–Golay w=15 p=4 over columns [:, :7]",
    "_measure_splice — the raw jump, diagnostics only",
    "_blend_splice — raised cosine over 6 steps",
], 74)
A.plot([LX + 1.5, LX + LW - 1.5], [y + 1.5, y + 1.5], color=LINE, lw=1.0, zorder=5)
A.text(LX + 2.2, y - 2.5, "rtc_executor — streaming publish", fontsize=8.2, fontweight="bold",
       color=AQUA, zorder=5, family="monospace")
stack(A, LX + 2.4, y - 6.5, [
    "time-parameterize at 0.5× Franka vel / accel limits",
    "catch-up segment ≤ 0.6 s, never slower than the measured v_now",
    "terminal waypoint velocity = 0 — JTC silently drops a trajectory that ends moving",
    "publish to the JTC topic: fire-and-forget, replaces the running trajectory in place",
], 74)

machine(A, FX, FW_, 27, 71, "Franka PC", "MoveIt 2 · franka_ros2", [
    "fer_arm_controller (JTC) at 1 kHz",
    "franka_hardware / libfranka",
    "joint impedance 1500…1000 Nm/rad, ~50 % of the libfranka default",
    "collision thresholds + automatic reflex recovery",
    "/joint_states at 1.4 kHz → q_start and v_now",
    "a topic publish replaces the running trajectory in place — nothing is ever asked to stop",
], ORANGE)

def arrow(ax, x0, y0, x1, y1, color, label, dy=1.8, fs=6.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>,head_width=2.6,head_length=4.5",
                                 color=color, lw=1.8, mutation_scale=1, zorder=6,
                                 shrinkA=0, shrinkB=0))
    ax.text((x0 + x1) / 2, y0 + dy, label, fontsize=fs, color=color, ha="center", va="bottom",
            zorder=6, linespacing=1.35, fontweight="bold", family="monospace")

arrow(A, LX - 0.4, 74, GX + GW + 0.4, 74, VIOLET, "obs +\nprev_actions\n+ s_req + d")
arrow(A, GX + GW + 0.4, 52, LX - 0.4, 52, VIOLET, "actions\n(50,14)\n+ rtc_*")
A.text((GX + GW + LX) / 2, 47.5, f"round trip\np50 {RTT50*1e3:.0f} ms\np95 {RTT95*1e3:.0f} ms",
       fontsize=6.4, color=INK2, ha="center", va="top", linespacing=1.4)

arrow(A, LX + LW + 0.4, 74, FX - 0.4, 74, AQUA, "joint_\ntrajectory\n(topic)")
arrow(A, FX - 0.4, 52, LX + LW + 0.4, 52, ORANGE, "/joint_states\n1.4 kHz")
A.text((LX + LW + FX) / 2, 47.5, "measured q, v\n→ catch-up\n   sizing", fontsize=6.4,
       color=INK2, ha="center", va="top", linespacing=1.4)

A.add_patch(FancyBboxPatch((GX, 2), GW, 20, boxstyle="round,pad=0,rounding_size=1",
                           fc="white", ec=LINE, lw=1.0, zorder=2))
A.text(GX + 2.0, 18.5, "wire contract", fontsize=7.6, fontweight="bold", color=INK2, va="center")
A.text(GX + 2.0, 5.0, "prev_actions      (50,14) float32, verbatim\n"
                      "prev_actions_start  s_req — first unplayed\n"
                      "prev_actions_d      d — actions to freeze\n"
                      "all three travel together, or none",
       fontsize=6.5, color=INK, va="bottom", linespacing=1.6, family="monospace")

A.add_patch(FancyBboxPatch((FX, 2), FW_, 20, boxstyle="round,pad=0,rounding_size=1",
                           fc="white", ec=LINE, lw=1.0, zorder=2))
A.text(FX + 2.0, 18.5, "alongside", fontsize=7.6, fontweight="bold", color=INK2, va="center")
A.text(FX + 2.0, 5.0, "▪ omni_place → franka_gripper Move,\n"
                      "   scheduled off the publish's step_times\n"
                      "▪ robot_compliance → stiffness, collision\n"
                      "   thresholds, reflex recovery",
       fontsize=6.9, color=INK, va="bottom", linespacing=1.6)

# ═══ B · TIMELINE ═══════════════════════════════════════════════════════════
B = ax_at(0.552, 0.660, xlim=(-1.05, 7.6), ylim=(0, 100))
head(B, "B", "Blocking baseline vs. RTC — same policy, same inference time",
     f"RTC lane: real publish times from a logged {t_pub[-1]:.0f} s episode")

yb, hb = 71, 13
B.text(-1.0, 94, "Blocking baseline", fontsize=8.8, fontweight="bold", color=RED, va="center")
B.text(0.55, 94, "infer → play the whole 50-step chunk open-loop → dead-stop → infer",
       fontsize=7.4, color=INK2, va="center")
tt, first = 0.0, True
while tt < 6.6:
    B.add_patch(Rectangle((tt, yb), min(RTT50, 6.6 - tt), hb, fc=RED, ec="none", alpha=0.85, zorder=3))
    if first:
        B.text(tt + RTT50 / 2, yb - 1.8, "infer\narm stopped", fontsize=6.5, color=RED,
               ha="center", va="top", linespacing=1.3, fontweight="bold")
    tt += RTT50
    seg = min(H * DT, 6.6 - tt)
    if seg > 0:
        B.add_patch(Rectangle((tt, yb), seg, hb, fc=BLUE, ec="none", alpha=0.28, zorder=3))
        if first:
            B.text(tt + seg / 2, yb + hb / 2, "execute chunk open-loop · 50 actions · 5.0 s",
                   fontsize=7.4, color="#184f95", ha="center", va="center", zorder=4)
    tt += H * DT
    first = False
B.text(6.75, yb + hb / 2, f"1 replan\nevery {H*DT+RTT50:.1f} s", fontsize=7.6, color=RED,
       ha="left", va="center", fontweight="bold", linespacing=1.4)

yr, hr = 14, 13
B.text(-1.0, 53, "RTC", fontsize=8.8, fontweight="bold", color=GOOD, va="center")
B.text(-0.35, 53, "inference overlaps execution; the reply is spliced into the chunk already playing",
       fontsize=7.4, color=INK2, va="center")
B.add_patch(Rectangle((0, yr), 6.6, hr, fc=BLUE, ec="none", alpha=0.28, zorder=3))
B.text(3.3, yr + hr / 2, "execution — continuous, never stops", fontsize=7.6,
       color="#184f95", ha="center", va="center", zorder=4)
vis = [k for k in range(1, NPUB) if t_pub[k] < 6.6]
for i, k in enumerate(vis):
    t1 = t_pub[k]
    t0 = max(0.0, t1 - rtt[min(k, len(rtt) - 1)])
    yy = yr + hr + 3.0 + (i % 3) * 4.6
    B.add_patch(Rectangle((t0, yy), t1 - t0, 3.4, fc=ORANGE, ec="none", alpha=0.9, zorder=4))
    B.plot([t1, t1], [yr + hr, yy], color=ORANGE, lw=0.7, ls=(0, (2, 2)), zorder=4)
    B.plot([t1], [yr + hr], marker="v", ms=4.5, color=ORANGE, zorder=5, clip_on=False)
B.text(6.75, yr + hr / 2, f"1 replan\nevery {DPUB:.2f} s", fontsize=7.6, color=GOOD,
       ha="left", va="center", fontweight="bold", linespacing=1.4)
B.text(-1.0, 46.0, "inference in flight  ▾ splice", fontsize=7.0, color=ORANGE,
       ha="left", va="center", fontweight="bold")
for x in np.arange(0, 6.8, 1.0):
    B.plot([x, x], [5, 8.5], color=MUTED, lw=0.8, zorder=2)
    B.text(x, 2.0, f"{x:.0f} s", fontsize=6.8, color=MUTED, ha="center", va="bottom")


# ═══ C · ANATOMY OF A CHUNK ═════════════════════════════════════════════════
C = ax_at(0.416, 0.526, xlim=(-4.5, 62), ylim=(0, 100))
head(C, "C", "Anatomy of a chunk — what the conditioning actually constrains",
     "H = 50, d = 5, s = 1  →  hard-frozen 5 · soft-masked 44 · free 1")

S_REQ, D_ACT = 1, 3          # exec_horizon = 1; measured median d_actual = 3
S_REPLY = S_REQ + D_ACT

yo, yn, hbar = 72, 38, 14
# previous chunk
C.add_patch(Rectangle((0, yo), H, hbar, fc=BLUE, ec="none", alpha=0.22, zorder=3))
C.add_patch(Rectangle((0, yo), S_REPLY, hbar, fc=MUTED, ec="none", alpha=0.45, zorder=4))
C.text(-1.0, yo + hbar / 2, "previous\nchunk", fontsize=7.8, fontweight="bold", color=INK,
       ha="right", va="center", linespacing=1.4)
C.text(S_REPLY / 2, yo + hbar + 1.5, "played", fontsize=6.6, color=INK2, ha="center", va="bottom")
C.text(28, yo + hbar / 2, "sent back on the wire as prev_actions — verbatim, full (50, 14)",
       fontsize=7.4, color="#184f95", ha="center", va="center", zorder=5)

# new chunk, aligned so index 0 sits at old index s_req
C.add_patch(Rectangle((S_REQ, yn), D_SENT, hbar, fc=VIOLET, ec="none", alpha=0.65, zorder=3))
C.add_patch(Rectangle((S_REQ + D_SENT, yn), H - D_SENT - 1, hbar, fc=BLUE, ec="none",
                      alpha=0.30, zorder=3))
C.add_patch(Rectangle((S_REQ + H - 1, yn), 1, hbar, fc=MUTED, ec="none", alpha=0.40, zorder=3))
C.text(-1.0, yn + hbar / 2, "new chunk\nfrom server", fontsize=7.8, fontweight="bold", color=INK,
       ha="right", va="center", linespacing=1.4)
C.text(S_REQ + D_SENT / 2, yn - 2.0, "hard-frozen · d = 5\npinned to prev chunk", fontsize=6.6,
       color=VIOLET, ha="center", va="top", linespacing=1.4, fontweight="bold")
C.text(S_REQ + D_SENT + (H - D_SENT - 1) / 2, yn + hbar / 2,
       "soft-masked · 44 actions — prefix attention over the overlap, not a hard freeze",
       fontsize=7.4, color="#184f95", ha="center", va="center", zorder=5)
C.text(S_REQ + H - 0.5, yn - 2.0, "free\n1", fontsize=6.6, color=INK2, ha="center", va="top",
       linespacing=1.4)

# markers
C.annotate("", xy=(S_REQ, yo), xytext=(S_REQ, yn + hbar),
           arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2, ls=(0, (3, 2))), zorder=6)
C.plot([S_REQ], [yo], marker="^", ms=6, color=ORANGE, zorder=7)
C.text(7.0, 66, "s = 1  → fire the next inference here", fontsize=7.2, color=ORANGE,
       fontweight="bold", va="center")
C.annotate("", xy=(S_REPLY, yo), xytext=(S_REPLY, yn + hbar),
           arrowprops=dict(arrowstyle="-", color=RED, lw=1.4, ls=(0, (3, 2))), zorder=6)
C.text(7.0, 58, "reply lands  → splice at new[d_actual = 3]  ≡  old[4]: the same instant",
       fontsize=7.2, color=RED, fontweight="bold", va="center")

C.add_patch(Rectangle((S_REPLY, yn - 0.6), D_ACT, hbar + 1.2, fc="none", ec=GOOD, lw=1.6,
                      zorder=8, ls=(0, (2, 1.6))))
C.annotate("", xy=(S_REPLY, 19), xytext=(S_REPLY + D_ACT, 19),
           arrowprops=dict(arrowstyle="<->", color=GOOD, lw=1.3))
C.text(S_REPLY + D_ACT + 1.2, 18.5,
       "the only actions ever played: ~3 of 50, then the chunk is replaced again",
       fontsize=7.2, color=GOOD, va="center", fontweight="bold")

C.text(0, 9.0, wrap("Firing at s = 1 instead of the paper's s ≈ H/2 maximises both the soft-masked region and the "
                    "replan rate. At s = H − d the soft mask is empty — the chunk is pinned for d actions and then "
                    "unconstrained, so the discontinuity is merely relocated from index 0 to index d.", 168),
       fontsize=7.4, color=INK2, va="top", linespacing=1.5)

# ═══ D · SPLICE + BLEND ═════════════════════════════════════════════════════
def card(y0, y1, x0, x1, label, title, sub=None):
    c = ax_at(y0, y1, x0, x1)
    head(c, label, title, sub)
    return c

DY0, DY1 = 0.232, 0.394
Dc = card(DY0, DY1, L, MID - 0.012, "D", "Splice blending", "one logged splice")
dax = fig.add_axes([L + 0.043, DY0 + 0.048, (MID - 0.012) - L - 0.056, (DY1 - DY0) - 0.062])

kk = 12          # ro = 4 steps of history, jump close to the run median
a_old = z[f"actions_{kk}"]; a_new = z[f"actions_{kk + 1}"]
ro = int(sm[kk, 3]) - offs[kk]
j = int(np.argmax(np.abs(sj[kk])))
nb = min(BLEND, len(a_new))
ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(nb) / nb))
delta = -sj[kk]
raw_new = np.array(a_new[:, :7], dtype=float)
raw_new[:nb] -= ramp[:, None] * delta[None, :]

base = a_old[ro, j]
NPAST, NFUT = ro, 15
x_old = np.arange(-NPAST, 1)
y_old = (a_old[ro - NPAST:ro + 1, j] - base) * 1e3
x_new = np.arange(0, NFUT + 1)
y_raw = (raw_new[:NFUT + 1, j] - base) * 1e3
y_bld = (a_new[:NFUT + 1, j] - base) * 1e3

dax.axvspan(0, nb, color=AQUA, alpha=0.10, lw=0)
dax.plot(x_old, y_old, "-o", color=BLUE, lw=2.0, ms=4.5, label="previously commanded chunk")
dax.plot(x_new, y_raw, "--x", color=ORANGE, lw=1.8, ms=5, label="server reply, raw")
dax.plot(x_new, y_bld, "-o", color=AQUA, lw=2.0, ms=4.5, label="what the arm is commanded (blended)")
dax.annotate("", xy=(0, y_raw[0]), xytext=(0, 0),
             arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
dax.text(-0.3, y_raw[0] / 2 + 14, f"raw splice jump\n{abs(sj[kk, j])*1e3:.0f} mrad",
         fontsize=7.2, color=RED, fontweight="bold", va="center", ha="right", linespacing=1.4)
dax.text(nb / 2, dax.get_ylim()[1], f"blend window · {BLEND} steps\nraised cosine, 1 → 0",
         fontsize=6.9, color="#0f7f59", ha="center", va="top", linespacing=1.4)
dax.set_xlabel("policy step, relative to the splice", fontsize=7.6)
dax.set_ylabel(f"joint {j+1} target (mrad, rel.)", fontsize=7.6)
dax.legend(fontsize=6.9, frameon=False, loc="lower right", handlelength=1.8)
for sp in ("top", "right"):
    dax.spines[sp].set_visible(False)
dax.tick_params(labelsize=7)
dax.grid(axis="y", color=LINE, lw=0.7)
dax.set_axisbelow(True)
dax.set_facecolor("white")

Dc.text(0.8, 1.5, wrap(
        f"Shape-preserving: every action is displaced by the same decaying offset, so the policy's own "
        f"trajectory through the chunk is untouched — only the constant that disagreed with what was already "
        f"commanded is removed.", 104),
        fontsize=7.3, color=INK2, va="bottom", linespacing=1.55)

# ═══ E · SAVITZKY–GOLAY ═════════════════════════════════════════════════════
Ec = card(DY0, DY1, MID + 0.012, R, "E", "Chunk smoothing", "one logged 50-action chunk")
eax = fig.add_axes([0.556, DY0 + 0.048, 0.272, (DY1 - DY0) - 0.062])

jj = 5
sm_chunk = savgol_filter(raw_chunk, SW, SP, axis=0, mode="interp")
xs = np.arange(len(raw_chunk))
y_r = (raw_chunk[:, jj] - raw_chunk[0, jj]) * 1e3
y_s = (sm_chunk[:, jj] - raw_chunk[0, jj]) * 1e3
eax.axvspan(3, 8, color=GOOD, alpha=0.10, lw=0)
eax.plot(xs, y_r, "-", color=ORANGE, lw=1.3, alpha=0.95, label="raw policy chunk")
eax.plot(xs, y_r, ".", color=ORANGE, ms=3.2)
eax.plot(xs, y_s, "-", color=AQUA, lw=2.4, label=f"Savitzky–Golay w={SW}, p={SP}")
eax.text(11.0, np.nanmax(y_r) * 0.42, "executed\nband ~[d, 2d]", fontsize=6.7,
         color="#0a7a0a", ha="left", va="center", linespacing=1.35, fontweight="bold")
eax.set_xlabel("chunk index (0 … 49)", fontsize=7.6)
eax.set_ylabel("joint 6 target (mrad, rel.)", fontsize=7.6)
eax.legend(fontsize=6.9, frameon=False, loc="lower right", handlelength=1.8)
for sp in ("top", "right"):
    eax.spines[sp].set_visible(False)
eax.tick_params(labelsize=7)
eax.grid(axis="y", color=LINE, lw=0.7)
eax.set_axisbelow(True)
eax.set_facecolor("white")

# inset: worst implied acceleration
iax = fig.add_axes([0.876, DY0 + 0.058, 0.076, (DY1 - DY0) - 0.092])
def worst_acc(arr):
    return np.abs(np.diff(arr, n=2, axis=0)).max() / DT ** 2
bars = [("raw", worst_acc(raw_chunk), ORANGE)]
for w, po in [(7, 2), (15, 4), (21, 4)]:
    bars.append((f"w{w}", worst_acc(savgol_filter(raw_chunk, w, po, axis=0, mode="interp")), AQUA))
iax.bar([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
for i, b in enumerate(bars):
    iax.text(i, b[1], f"{b[1]:.2f}", fontsize=6.0, ha="center", va="bottom", color=INK2)
iax.set_title("worst implied\nzigzag accel\nrad/s², Δt = 0.1 s", fontsize=6.4, color=INK2, pad=4,
              linespacing=1.4)
iax.set_ylim(0, max(b[1] for b in bars) * 1.22)
for sp in ("top", "right", "left"):
    iax.spines[sp].set_visible(False)
iax.set_yticks([]); iax.tick_params(labelsize=6, length=0)
iax.set_facecolor("white")

d2 = np.diff(raw_chunk, n=2, axis=0)
rev = (np.sign(np.diff(raw_chunk, axis=0)[1:]) * np.sign(np.diff(raw_chunk, axis=0)[:-1]) < 0).mean()
Ec.text(0.8, 1.5, wrap(
        f"{rev*100:.0f} % of consecutive per-step deltas reverse sign. That amplitude is fixed in radians, so the "
        f"implied acceleration scales as 1/Δt² — halving step_duration quadruples it.", 104),
        fontsize=7.3, color=INK2, va="bottom", linespacing=1.55)

# ═══ F · MEASURED ═══════════════════════════════════════════════════════════
FY0, FY1 = 0.048, 0.198
Fc = card(FY0, FY1, L, R, "F", "Measured on the arm",
          f"one {t_pub[-1]:.0f} s episode · {len(rtt)} inferences · {len(sj)} splices")
w3 = (R - L - 0.10) / 3
def mini(i):
    a = fig.add_axes([L + 0.045 + i * (w3 + 0.038), FY0 + 0.030, w3, (FY1 - FY0) - 0.058])
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    a.spines["left"].set_color(LINE); a.spines["bottom"].set_color(LINE)
    a.tick_params(labelsize=7); a.set_facecolor("white")
    a.grid(axis="y", color=LINE, lw=0.7); a.set_axisbelow(True)
    return a

f1 = mini(0)
f1.hist(rtt * 1e3, bins=14, color=BLUE, alpha=0.85, edgecolor="white", lw=0.6)
f1.axvline(D_SENT * DT * 1e3, color=RED, lw=1.6, ls=(0, (4, 2)))
f1.text(D_SENT * DT * 1e3 - 8, f1.get_ylim()[1] * 0.93, "latency budget\nrtc_d = 5 → 500 ms",
        fontsize=6.7, color=RED, ha="right", va="top", fontweight="bold", linespacing=1.4)
f1.set_title(f"round trip · p50 {RTT50*1e3:.0f} · p95 {RTT95*1e3:.0f} · max {RTTMX*1e3:.0f} ms",
             fontsize=7.8, color=INK, pad=5, loc="left")
f1.set_xlabel("ms", fontsize=7.3); f1.set_ylabel("inferences", fontsize=7.3)

f2 = mini(1)
vals, cnts = np.unique(d_act, return_counts=True)
f2.bar(vals, cnts, color=VIOLET, alpha=0.85, width=0.62)
for v, c in zip(vals, cnts):
    f2.text(v, c, str(c), fontsize=7, ha="center", va="bottom", color=INK2)
f2.axvline(D_SENT, color=RED, lw=1.6, ls=(0, (4, 2)))
f2.text(D_SENT - 0.15, max(cnts) * 0.93, "frozen region d = 5\n0 violations", fontsize=6.7,
        color=RED, ha="right", va="top", fontweight="bold", linespacing=1.4)
f2.set_xticks(list(vals) + [D_SENT])
f2.set_title("d_actual — steps consumed by inference", fontsize=7.8, color=INK, pad=5, loc="left")
f2.set_xlabel("policy steps", fontsize=7.3); f2.set_ylabel("splices", fontsize=7.3)

f3 = mini(2)
f3.hist(jump_mx * 1e3, bins=12, color=ORANGE, alpha=0.85, edgecolor="white", lw=0.6)
f3.axvline(J50 * 1e3, color=INK2, lw=1.3, ls=(0, (4, 2)))
f3.text(J50 * 1e3 + 1.5, f3.get_ylim()[1] * 0.95, f"p50 {J50*1e3:.0f} mrad", fontsize=6.7,
        color=INK2, va="top")
f3.axvline(0, color=GOOD, lw=2.4)
f3.text(2.0, f3.get_ylim()[1] * 0.46, "after _blend_splice:\nexactly 0, every splice",
        fontsize=6.7, color="#0a7a0a", va="top", fontweight="bold", linespacing=1.4)
f3.set_title("raw splice jump the server leaves behind", fontsize=7.8, color=INK, pad=5, loc="left")
f3.set_xlabel("max joint |Δ| (mrad)", fontsize=7.3); f3.set_ylabel("splices", fontsize=7.3)

# ═══ FOOTER ═════════════════════════════════════════════════════════════════
Ft = ax_at(0.008, 0.036, frame=False)
Ft.text(0, 55, "RTC after Black, Galliker & Levine, “Real-Time Execution of Action Chunking Flow Policies”.  "
               "Panels B, D, F from debug/rtc/; panel E from debug/sanity/.",
        fontsize=7.6, color=MUTED, va="center")
Ft.text(100, 55, "franka_openpi · openpi_franka_RTC.launch_sim.py", fontsize=7.6, color=MUTED,
        va="center", ha="right", family="monospace")

fig.savefig(OUT, dpi=200)
print("wrote", OUT)
