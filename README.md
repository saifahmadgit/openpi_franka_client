# franka_openpi

ROS 2 package that runs an [OpenPI](https://github.com/Physical-Intelligence/openpi) pi0.5 policy on a **Franka FER / Panda** arm.

Two deployment pipelines live here:

| Pipeline | Node | Executor | Behaviour |
|---|---|---|---|
| **Blocking** (baseline) | `openpi_client_node` | `action_executor` (FollowJointTrajectory **action**) | Waits for inference, executes the chunk, stops, waits again. Arm dead-stops at every chunk boundary. |
| **RTC** (what you should use) | `openpi_rtc_node` | `rtc_executor` (JTC **topic**) | Fires the next inference *while the arm is still moving*, conditions it on the chunk being executed, and splices the reply in. No stops. |

Everything below is about the RTC path unless stated otherwise.

> **Never run both pipelines at once.** They both drive `fer_arm_controller`. A topic
> publish from `rtc_executor` yanks the trajectory out from under an active action goal.

---

## TL;DR — the command that works

This is the tuned configuration for the sim-trained checkpoint on the real arm:

```bash
ros2 launch franka_openpi openpi_franka_RTC.launch_sim.py \
  prompt:="pick up the mustard bottle" \
  step_duration:=0.1 \
  rtc_d:=5 \
  exec_horizon:=1 \
  rtc_d_margin:=3 \
  send_prev_actions:=true \
  auto_error_recovery:=true \
  splice_blend_steps:=6 \
  smooth_window:=15 \
  smooth_polyorder:=4 \
  joint_stiffness:="[1500.0, 1500.0, 1500.0, 1250.0, 1250.0, 1000.0, 1000.0]"
```

Why these values, in one line each:

* `step_duration:=0.1` — 10 Hz policy cadence. Twice the old 0.2 default; the smoothing/blend/catch-up work below is what made that survivable.
* `exec_horizon:=1` — replan on **every** inference. The chunk is republished as soon as a reply lands, so only actions `[d, 2d]` of each chunk are ever played and the loop is as closed as latency allows.
* `rtc_d:=5` — fixed latency budget of 5 steps (500 ms at 0.1 s/step), covering the measured p95 round trip.
* `rtc_d_margin:=3` — how far `d` ratchets up when a deadline is missed.
* `splice_blend_steps:=6` — 0.6 s raised-cosine decay of the residual splice offset the server leaves unpinned.
* `smooth_window:=15 smooth_polyorder:=4` — Savitzky-Golay over each chunk, killing the policy's step-to-step zigzag.
* `joint_stiffness:=[1500…1000]` — ~50 % of libfranka default. Stiffer than the file default `[600…50]` because at 10 Hz you need the arm to actually track; still soft enough to yield on contact.

Prerequisites for that command are three things running elsewhere: the **policy server** on the GPU box, **MoveIt 2 + franka driver** on the Franka PC, and **omni_place** on the laptop. See [Running](#running).

---

## Repos you need

| Repo | Where | Why |
|---|---|---|
| **this repo** (`franka_openpi`) | laptop, `~/Franka/src/` | The client, the executors, the launch files. |
| [`franka-vision-guided-manipulation`](https://github.com/saifahmadgit/franka-vision-guided-manipulation) | laptop, same workspace | Provides `omni_place` → `MotionPlanningInterface`, imported by `action_executor` for the **gripper** (`set_gripper_franka`). Build it or nothing imports. |
| [`openpi`](https://github.com/Physical-Intelligence/openpi) (or the fork [`openpi_franka`](https://github.com/saifahmadgit/openpi_franka)) | GPU server | `scripts/serve_policy.py` — the websocket policy server. **Must be the RTC-capable build** (see [Server contract](#server-contract)). |
| `openpi-client` (pip package from the same repo) | laptop `.venv` | `websocket_client_policy`, `msgpack_numpy`, `image_tools`. |
| [`franka_ros2`](https://github.com/frankaemika/franka_ros2) / `franka_description` | Franka PC + laptop | `fer_arm_controller`, `franka_msgs`, `franka_hardware` service/action servers, `franka_robot_state_broadcaster`. |
| `ros-jazzy-realsense2-camera` | laptop | Three RealSense streams. |
| `scipy`, `numpy`, `opencv-python`, `cv_bridge`, `matplotlib` | laptop | `savgol_filter` is a hard import in `openpi_rtc_node`. |

### Install

```bash
cd ~/Franka/src
git clone https://github.com/saifahmadgit/openpi_franka_client franka_openpi
git clone https://github.com/saifahmadgit/franka-vision-guided-manipulation

cd ~/Franka
python3 -m venv .venv
.venv/bin/pip install openpi-client scipy numpy

colcon build --symlink-install
source install/setup.bash
```

The launch files prepend `~/Franka/.venv/lib/python3.12/site-packages` to `PYTHONPATH`
so the system-Python ROS entry point can import `openpi_client`. If you moved the
workspace or changed Python versions, fix `_VENV_SITE` at the top of the launch file.

---

## Machine layout

```
GPU server (129.105.69.11)              Laptop                                Franka PC
─────────────────────────────           ──────────────────────────────        ─────────────────
 serve_policy.py :8000                   3× RealSense  ──┐
   pi0.5 checkpoint                                      │
        ▲                                openpi_rtc_node ┤ packs obs
        │  obs + prev_actions ───────────────────────────┘  {cam_high, cam_left_wrist,
        │  (websocket, msgpack)                              cam_right_wrist, state(9), prompt,
        │                                                    prev_actions, _start, _d}
        └── actions (H,14) + rtc_* flags ──►  _smooth_chunk  (Savitzky-Golay)
                                              _blend_splice  (raised cosine)
                                                    │
                                             rtc_executor
                                              time-parameterize + publish
                                                    │  /fer_arm_controller/joint_trajectory
                                                    ▼                          ROS 2 DDS
                                                                        fer_arm_controller (JTC)
                                              omni_place ── gripper ──►  franka_hardware / libfranka
                                              robot_compliance ──────►  stiffness / collision / recovery
```

---

## RTC — what it actually does

From Black, Galliker & Levine, *Real-Time Execution of Action Chunking Flow Policies*.
A chunked policy normally: infer → execute all H actions → stop → infer. The stop is
the inference latency, and it is visible as a hitch at every chunk boundary.

RTC removes it with three moves, all of which live in `openpi_rtc_node.py`:

1. **Never block on inference.** `rtc_executor` publishes to the controller's *topic*
   interface, which is fire-and-forget; JTC replaces whatever trajectory is running.
   The arm keeps playing the current chunk while the request is in flight.
2. **Fire early.** Request the next chunk at chunk index `s` (`exec_horizon`), not at H.
3. **Condition on what is already executing.** Send the previous chunk back with the
   request, plus the index of the first action not yet executed, so the server samples
   a chunk that *agrees* with the motion already committed.

### The three regions

Given horizon `H`, execution horizon `s`, latency `d`:

```
new chunk:  [0 ────── d) [d ────────── H-s) [H-s ────── H)
             hard-frozen   soft-masked        free
             pinned to     prefix-attends     unconstrained
             prev chunk    to prev chunk
```

`_mask_regions()` prints these in the end-of-episode report. **If `soft-masked == 0`,
RTC is doing nothing useful** — the chunk is pinned for `d` actions then unconstrained,
so the discontinuity just relocates from index 0 to index `d`. That happens when
`s = H - d` (i.e. `exec_horizon:=0`), which is kept only for comparison runs.

With `exec_horizon:=1` and `H=50, d=5`: hard 5, soft 44, free 1. Maximum conditioning,
maximum replan rate. The paper's `s ≈ H/2` is a compute-budget compromise; if your
server keeps up, small `s` is strictly better closed-loop behaviour.

### The cycle, step by step

```
 while steps_done < max_steps:
   ├─ if robot in REFLEX → recover, clear RTC state, re-prime, continue
   ├─ 1. sleep until _chunk_index() reaches fire_at = min(exec_horizon, H-d)
   ├─ 2. s_req = _chunk_index();  fire infer(obs + prev_actions, prev_actions_start=s_req,
   │                                          prev_actions_d=d_sent) on a worker thread
   ├─ 3. poll until the reply lands; if the chunk runs dry first → DROPPED DEADLINE,
   │     ratchet _d_floor = d_sent + rtc_d_margin
   ├─ 4. s_reply = _chunk_index();  d_actual = s_reply - s_req
   │     if d_actual > d_sent → FROZEN REGION EXCEEDED, ratchet _d_floor
   ├─ 5. _measure_splice()  (diagnostics, on the RAW server reply)
   ├─ 6. _blend_splice()    (fix the residual discontinuity)
   └─ 7. _publish(new_chunk, offset=d_actual)
```

`_chunk_index()` maps wall time onto a chunk index through the **`step_times` the
executor computed for this publish**, not through `elapsed / step_duration` — velocity
limiting stretches segments, so the naive version drifts.

### Server contract

The client sends, on every non-priming call:

| Key | Type | Meaning |
|---|---|---|
| `prev_actions` | `(H, 14) float32` | The previously commanded chunk, **verbatim**. |
| `prev_actions_start` | `int` | Index of the first action not yet executed (`s_req`). |
| `prev_actions_d` | `int` | How many actions the server should freeze. |

All three travel together or not at all — the server sees a complete context or none.

The reply should carry `rtc_active`, `rtc_d_eff`, `rtc_start`, `rtc_overlap`,
`rtc_prefix_attention_horizon`. `_record_rtc_flags` logs them once and **errors loudly
if `rtc_active=False` while `send_prev_actions:=true`** — that means the server is
ignoring your conditioning and every splice is a raw discontinuity between two
independent samples.

> ⚠️ **The trap.** The config uses `use_delta_joint_actions`: the model predicts offsets
> from the state sent on that call, and the server adds the state back before replying.
> So the reply is stored **untouched at full (H, 14) width** and sent back untouched — no
> slicing, no delta conversion, no rebasing. Rebasing here would pin the arm to targets
> stale by exactly the motion RTC exists to bridge. Only the execution path takes
> `[:, :8]`. The server owns the action space; this client never reasons about it.

### Measured reality of this server

Probed directly at `129.105.69.11:8000` with `prev_actions_d=18`: the reply says
`rtc_active=True, rtc_d_eff=18, rtc_prefix_attention_horizon=38`, but `new[0:18]` still
differs from `prev[start:start+18]` by up to **0.030 rad**. (Independent samples differ
by 0.083, so conditioning *is* working.) It is soft **prefix attention** over the
overlap, not a hard freeze. That leftover is what `splice_blend_steps` exists to absorb.

---

## Smoothing, splicing, and why they matter more as you speed up

Both problems below are **fixed amplitudes in radians**. Halving `step_duration` does
not shrink them — it doubles the velocity step and quadruples the acceleration. This is
the entire reason the 0.2 → 0.1 s move needed new machinery.

### 1. Chunk smoothing — `smooth_window` / `smooth_polyorder`

The policy's chunks zigzag at the per-action level: measured over 20 logged chunks,
**27 % of consecutive per-step deltas reverse sign**, second-difference amplitude
p50 7.4 / p95 26 mrad.

| step_duration | implied zigzag acceleration |
|---|---|
| 0.15 s | 0.33 rad/s² — invisible |
| 0.075 s | 4.66 rad/s² p95, 13.0 max — **exceeds `ACC_LIMIT`**, felt as vibration |

`_smooth_chunk` runs `scipy.signal.savgol_filter` over columns `[:, :7]` of every chunk
as it arrives — **priming included**, so consecutive chunks are filtered identically.
An asymmetry here would itself show up as a step at the splice. Column 7 (gripper) is
deliberately *not* filtered: it is thresholded into binary open/close, so smoothing it
only moves the crossing instant. Columns 8–13 are padding.

**Window sizing is governed by which part of the chunk is executed, not the whole chunk.**
With `exec_horizon:=1`, a chunk is published at `offset=d_actual` and replaced
`d_actual` steps later, so only indices ~`[d, 2d]` are ever played — measured
`d_actual` p50 3 / max 7, i.e. a band around indices 3–8 of 50. Savitzky-Golay's
distortion is concentrated at the array **edges** (`mode="interp"` polyfits there), and
those indices are never reached:

| filter | worst zigzag accel | EE dev, whole chunk | EE dev, executed band |
|---|---|---|---|
| w=7, p=2 | 2.05 rad/s² | 4.03 mm max | 1.39 / 1.54 mm |
| w=15, p=3 | 0.91 rad/s² | 5.65 mm max | 1.43 / 1.61 mm |
| w=21, p=4 | 0.64 rad/s² | 5.59 mm max | 1.44 / 1.62 mm |

The band figure is flat across every window, so a wider window buys smoothness
essentially for free — 2.3× smoother for 0.04 mm. Beyond ~15 the filter spans more than
1.5 s of a 5 s chunk and starts blunting real motion.

> **If you raise `exec_horizon` above ~5, re-check this.** The executed band moves toward
> the chunk edges, where the distortion actually lives.
>
> If a grasp regresses, drop to `smooth_window:=7 smooth_polyorder:=2` for the
> least-filtered baseline. `smooth_window:=0` or `1` disables it entirely.

### 2. Splice blending — `splice_blend_steps`

`old[s_reply]` and `new[d_actual]` name the **same instant** — the first action commanded
from the new chunk is the one replacing `old[s_reply]`. Their difference is the step
change in commanded target the arm sees at the handover.

Measured splice jump p50: **0.0226 rad at step_duration 0.15, 0.0219 rad at 0.075** —
identical, as predicted, because the residual is in radians. At 0.15 s that is a
0.15 rad/s velocity step; at 0.075 s the same 0.022 rad is 0.30 rad/s and 4× the
acceleration, once per cycle. That is what reads as "lost smoothness at speed".

`_blend_splice` adds the difference back at the splice index and tapers it to zero over
`splice_blend_steps` actions with a **raised cosine** (1.0 at the splice → 0.0), not a
linear ramp — a linear ramp leaves a slope discontinuity at both ends, which is the same
problem one derivative up.

It is **shape-preserving**: every action in the window is displaced by the same decaying
offset, so the policy's own trajectory through the chunk is untouched. Only the constant
that made it disagree with what was already commanded is removed. The gripper column is
left alone.

Order matters: `_measure_splice` runs **before** the blend, so the diagnostics keep
reporting the raw jump the server hands back — that is the number telling you whether
RTC conditioning works. The **blended** array is what the arm executes *and* what goes
back on the wire next cycle, since `prev_actions` means "previously commanded".

`splice_blend_steps:=0` publishes the server chunk verbatim.

### 3. Catch-up segment — `CATCHUP_MAX_SEC` (constant in `rtc_executor.py`)

Segment 0 of every published trajectory spans *measured position → first commanded
target*, so it carries accumulated tracking error, not one policy step of motion —
measured at ~4× a normal step. Giving it a plain `step_duration` makes the arm cover it
at 4× the policy's speed: a lurch at every republish followed by a crawl.

`_time_parameterize` instead sizes it by distance at the chunk's own nominal speed,
capped at `CATCHUP_MAX_SEC = 0.6 s`. **That cap is absolute, not a multiple of
step_duration** — it used to be 4 × step_duration, which silently halved the catch-up
budget every time you sped up, exactly backwards. At 0.075 s, 5 of 16 publishes were
pinned at the old cap, forcing the catch-up segment above normal speed.

There is a second guard: the catch-up is never commanded **slower than the arm is already
moving** (`v_now` from `/joint_states`). Replanning from measured position while ignoring
measured velocity asks the arm to brake on 76 % of publishes (commanded/actual speed p50
0.89, p10 0.58) — an event-triggered average over the 1.4 kHz joint log shows
−0.118 rad/s at t=+60 ms after every publish, ringing at ~4 Hz. Capping `t_catch` at the
time the gap takes at the current speed removes the step, and only ever *shortens*
segment 0, so the vel/accel iteration still bounds the result.

### 4. Terminal velocity must be zero

`_waypoint_velocities` zeroes the velocity of the final waypoint, and **must**.
Measured on hardware, same 50 points, same ramp, only the last element differing:

```
terminal |vel| = 0.000  →  joint 7 moved +0.0706 of 0.080 commanded
terminal |vel| = 0.008  →  joint 7 moved +0.0004   (nothing at all)
```

`fer_arm_controller` silently discards a trajectory that ends moving. Baking a stop into
the chunk tail costs nothing here because the tail is **replaced before the arm reaches
it** — that is what RTC buys.

---

## Every parameter

### RTC core

| Param | Default (sim launch) | Tuned | What it means |
|---|---|---|---|
| `step_duration` | `0.2` | `0.1` | Seconds per policy step. Dataset is 30 fps (0.0333); this is the knob that sets arm speed. Everything above scales badly against it. |
| `exec_horizon` (`s`) | `25` | `1` | Chunk index at which the next inference fires. Capped internally at `H-d`. `0` = fire at `H-d` (degenerate: empty soft mask). Low = replan faster. |
| `rtc_d` | `3` | `5` | Inference latency in control steps; the server freezes this many actions. `0` = adapt from observed round trips (`max(observed) + rtc_d_margin`). |
| `rtc_d_margin` | `2` | `3` | Steps added on top of the worst observed `d` when the floor ratchets. Pure safety margin — the splice index must never exceed the frozen region. |
| `send_prev_actions` | `true` | `true` | `false` streams chunks with no conditioning, to separate "non-blocking streaming fixed it" from "RTC conditioning fixed it". |
| `max_steps` | `500` | — | Policy steps *played* before the episode is cut off (counts played steps, not splice offsets). |
| `num_episodes` | `1` | — | Episodes per run. RTC state is fully cleared between them. |
| `prompt` | `"pick up the red object"` | your task | Language instruction sent with every observation. |

### Smoothing / splicing

| Param | Default | Tuned | What it means |
|---|---|---|---|
| `smooth_window` | `15` | `15` | Savitzky-Golay window over the 7 joint columns. `0`/`1` = off. Even values are decremented; clamped to chunk length. |
| `smooth_polyorder` | `3` | `4` | SG polynomial order. Filter is skipped if `window <= polyorder`. |
| `splice_blend_steps` | `10` | `6` | Steps over which the residual splice offset decays (raised cosine). `0` = publish verbatim. |

### Images / transport

| Param | Default | What it means |
|---|---|---|
| `image_size` | `0` | `0` = send full resolution, server resizes (current, known-good). `224` pre-resizes with `resize_with_pad`, cutting the payload **2.77 MB → 0.45 MB** — most of the measured round trip. **Only safe if the server resizes to the same size with the same function**, in which case its own call short-circuits. If the server's transform differs, you stack two resamples and silently shift the input distribution. Confirm server-side first. |
| `two_front_cameras` | `true` | `true` = 3-camera mode (`cam_high`, `cam_left_wrist`, `cam_right_wrist`). `false` = 2-camera. |
| `server_host` / `server_port` | `129.105.69.11` / `8000` | Hardcoded in the launch file. Note **8000**, not the 8001 the non-RTC launches use — 8000 is what `serve_policy.py` actually listens on. |

### Gripper (sim RTC launch only)

| Param | Default | What it means |
|---|---|---|
| `gripper_open_width` | `0.08` | Aperture when open, m (full stroke is 0.08). **The biggest lever**: every open/close pays for the travel between open and close width, so narrowing this to just clear the object shortens *both* directions. 0.065 on a ~50 mm object takes the close to 0.45 s. |
| `gripper_close_width` | `0.02` | Aperture when closed, m. |
| `gripper_open_speed` | `0.1` | m/s of **width** (≈ half that per finger). 0.1 is the Franka Hand's rated max; more is clamped. |
| `gripper_close_speed` | `0.1` | m/s of width. Was 0.05, making the close 1.2 s against the open's 0.6 s for identical travel. This path uses `Move` with `adaptive_stop=False`, so faster close meets the object harder — **lower this first if contact trips the reflex**. |

Transitions are binary: `action[7] >= GRIPPER_CLOSE_THRESHOLD (0.02)` → open, else close.
They are scheduled off the publish's `step_times`, and any transitions pending from a
replaced chunk are cancelled along with the trajectory.

### Compliance (sim RTC launch only)

| Param | Default | Tuned | What it means |
|---|---|---|---|
| `joint_stiffness` | `[600,600,600,600,250,150,50]` | `[1500,1500,1500,1250,1250,1000,1000]` | libfranka internal joint-impedance stiffness, Nm/rad. libfranka default is `[3000,3000,3000,2500,2500,2000,2000]`; the file default is ~20 % of that (very soft, yields on contact), the tuned value is ~50 % (tracks well enough for 10 Hz while still yielding). `"[]"` leaves the robot's current setting alone. |
| `collision_torque` | `[40,40,38,36,32,28,24]` | — | Per-joint reflex torque thresholds, Nm. Fills nominal **and** acceleration, upper **and** lower. Raise to tolerate more contact. |
| `collision_force` | `[40,40,40,50,50,50]` | — | Cartesian reflex force/torque thresholds (x y z r p y). |
| `auto_error_recovery` | `true` | `true` | Clear a reflex and re-prime instead of stalling. **Keep this on**: the arm ignores commands in `ROBOT_MODE_REFLEX`, and the JTC topic interface is fire-and-forget, so without it one collision silently wastes every remaining step of the episode. |

Compliance is applied once at node startup, after the controller is up but before any
motion. These are **runtime settings on the robot, not ROS parameters** — they are lost
on every `franka_hardware` restart, which is why they are re-pushed each launch. All
three servers live on `franka_hardware` nodes, so with mock hardware every entry point
degrades to a warning and a no-op rather than blocking startup.

Reflex recovery clears the fault, then **discards RTC state and re-primes from scratch**
— the trajectory was dropped when the reflex hit and the arm is wherever it stopped, not
at the index `step_times` imply. Splicing against that stale chunk would splice against
actions the arm never reached.

### Velocity / acceleration limiting (constants in `action_executor.py`)

```python
ENFORCE_LIMITS = True
_LIMIT_SCALE   = 0.5                     # conservative fraction of Franka's official limits
VEL_LIMIT = [2.175]*4 + [2.61]*3  × 0.5  # rad/s
ACC_LIMIT = [15, 7.5, 10, 12.5, 15, 20, 20] × 0.5  # rad/s²
```

Segment durations only ever **stretch**, never shrink below `step_duration`, so limiting
can slow motion but never speed it past the trained cadence.

---

## Running

### 1. Policy server (GPU box)

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/serve_policy.py \
  --default_prompt "pick up the mustard bottle" \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_magicsim_apple_red \
  --policy.dir checkpoints/pi05_magicsim_apple_red/magicsim_apple_lora_red/55000
```

`XLA_PYTHON_CLIENT_PREALLOCATE=false` stops JAX grabbing all GPU memory upfront.
Port **8000** — that is what the RTC launch files connect to.

### 2. Franka PC

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch franka_moveit_config moveit.launch.py robot_ip:=<ROBOT_IP>
```

That is the only thing running there. It brings up `fer_arm_controller`,
`franka_hardware`'s service/action servers, and `franka_robot_state_broadcaster`.

### 3. Laptop — terminal 1: MoveIt / gripper interface

```bash
source ~/Franka/install/setup.bash
ros2 launch omni_place omni_place.launch.py
```

### 4. Laptop — terminal 2: cameras + RTC client

```bash
source ~/Franka/install/setup.bash
ros2 launch franka_openpi openpi_franka_RTC.launch_sim.py <args from TL;DR>
```

### Which launch file

| File | Checkpoint | Camera serials | RTC params exposed |
|---|---|---|---|
| `openpi_franka_RTC.launch_sim.py` | **sim-trained** | front_1/front_2 **swapped** | all of them (smoothing, splice, gripper, compliance) |
| `openpi_franka_RTC.launch.py` | real-data | dataset order | core RTC only — no smoothing/splice/gripper/compliance args |
| `openpi_franka.launch_sim.py` | sim-trained | swapped | blocking baseline |
| `openpi_franka.launch.py` | real-data | dataset order | blocking baseline |

If you need the smoothing/compliance knobs on the real-data checkpoint, copy the
`DeclareLaunchArgument` blocks and the matching `LaunchConfiguration` entries across —
the node declares all of them regardless, so the non-sim launch just falls back to node
defaults today.

---

## Cameras

Three RealSense D4xx at 640×480×30, colour only (depth/IR/IMU disabled).

| Node namespace | Topic | RTC **sim** serial | RTC **real** serial | Server key (3-cam) | Server key (2-cam) |
|---|---|---|---|---|---|
| `camera/front_1` | `/camera/front_1/camera/color/image_raw` | `342522070195` | `938422076779` | `cam_left_wrist` | `cam_high` |
| `camera/front_2` | `/camera/front_2/camera/color/image_raw` | `938422076779` | `342522070195` | `cam_right_wrist` | — |
| `camera/wrist` | `/camera/wrist/camera/color/image_raw` | `347622076595` | `347622076595` | `cam_high` | `cam_left_wrist` |

> **Do not "fix" the mapping by name.** `cam_high ← wrist` looks wrong and is correct:
> it matches how training filled the keys (`cam_high` is the mandatory `base_0_rgb`
> slot). The sim launch additionally swaps front_1/front_2 because the sim checkpoint's
> `front_1` key learned the physically-**right**-side view. `tools/openpi_replay.py
> --swap-front` settles which mapping is right offline, against training frames, with no
> robot involved.

`serial_no` must be wrapped in `ParameterValue(..., value_type=str)` — a bare numeric
string gets coerced to an int and the camera node rejects it.

`camera_viewer` is launched alongside for a live look at what is being sent.

---

## Wire format

**Observation → server**

```python
{
  "state":  np.float32[9],          # fer_joint1..7 + finger_joint1 + finger_joint2
  "images": {                       # CHW uint8, full resolution unless image_size > 0
      "cam_high":        ...,
      "cam_left_wrist":  ...,
      "cam_right_wrist": ...,       # 3-camera mode only
  },
  "prompt": str,
  # non-priming calls only:
  "prev_actions": np.float32[H, 14],
  "prev_actions_start": int,
  "prev_actions_d": int,
}
```

State is **9-dim, not 8** — the checkpoint's `norm_stats` are 9-dim and sending 8
mis-aligns normalization.

**Actions ← server**

```
(H, 14) float32, absolute joint targets after the server re-adds state
  [:, 0:7]   arm joint targets (rad)      ← smoothed, blended, executed
  [:, 7]     gripper                       ← thresholded at 0.02, binary
  [:, 8:14]  padding                       ← untouched, kept for the wire round-trip
```

H is typically 50.

---

## Diagnostics

### End-of-episode report

```
RTC timing
  round trip  p50 ... ms   p95 ... ms   max ... ms  (N calls)
  server_timing.infer_ms  p50 ...  p95 ...   <- real model time
  policy_timing.infer_ms  p50 ...          (JAX dispatch only -- reads far low)
  d observed  p50 ...  max ... steps  (d sent ..., firing at index ... of ...)
  RTC regions  hard-frozen ...  soft-masked ...  free ...
  splice jump  p50 ...  p95 ...  max ... rad  (N splices)
    per-joint mean |jump|: [...]
  dropped deadlines: N
  collision reflexes recovered: N
  frozen-region violations: N
```

Read it like this:

* **`policy_timing` ≪ `server_timing`** is expected, not a bug. `policy_timing.infer_ms`
  times JAX *dispatch*; JAX is async and the computation doesn't block until the result
  is converted back to numpy, outside that timer. `server_timing.infer_ms` wraps the
  whole `policy.infer` and is the honest number.
* **`soft-masked == 0`** → lower `exec_horizon`. RTC is relocating the discontinuity, not
  removing it.
* **`splice jump`** is the step change in commanded target at each handover, measured on
  the **raw** server reply. This is the headline RTC metric — it should fall toward 0 as
  the server's freezing improves. Compare it before/after any server-side change.
* **`dropped deadlines > 0`** → the chunk ran dry with inference still in flight. The arm
  holds its last action (JTC holds the final point when a trajectory expires — a stall,
  not a fault). Raise `rtc_d`, lower `exec_horizon`, or shrink the payload (`image_size`).
* **`frozen-region violations > 0`** → spliced at `d_actual > d_sent`, i.e. resumed inside
  actions the server never pinned. Each one is a visible jerk. `_d_floor` ratchets
  permanently for the rest of the run, so this self-heals, but the first one is felt.
* **`collision reflexes`** → lower `joint_stiffness` or raise `collision_torque`. If it is
  the *gripper* meeting the object, lower `gripper_close_speed` first.

### Debug files

Written to `~/Franka/src/franka_openpi/debug/rtc/` on exit, prefixed by `debug_tag`
(`rtc_sim_` for the sim launch, `rtc_` for the real one):

| File | Contents |
|---|---|
| `*_actual.npy` / `*_actual_times.npy` | Measured joints `[:8]`, logged at the driver's **native ~1.4 kHz** rate from `_joint_cb`, with monotonic timestamps. Deliberately not sampled on a fixed tick — a `step_duration` tick at 0.05 s has a 10 Hz Nyquist and aliases away the entire 10–50 Hz band vibration lives in. |
| `*_commanded.npz` | One entry per publish: `t_k`, `offset_k`, `step_times_k`, `actions_k`. Ragged (the tail shortens as `d` grows), hence npz. Includes tail steps that were later overwritten — the analysis needs both what was commanded and what was actually reached. |
| `*_rtt.npy` | Every round trip, including the JIT-warm first call (the report drops it). |
| `*_splice_jumps.npy` / `*_splice_meta.npy` | Per-joint jump `(n, 7)` and `(s_req, s_reply, d_actual, i_old, i_new)`. |

Nothing is plotted during the run. The blocking pipeline rendered two figures inside its
control loop, measured at 1.3–1.5 s of dead time per chunk.

### Tools

All of these are robot-safe — none creates an action client, so none can move the arm.

```bash
# 1. Pure logic check: index math, splice offsets, verbatim wire format.
#    Stubs out ROS and the websocket entirely. No robot, no server.
python3 tools/rtc_logic_check.py

# 2. Round-trip latency probe -> pick rtc_d. Uses live frames if cameras are up,
#    --synthetic (identical payload size) if not.
PYTHONPATH=~/Franka/.venv/lib/python3.12/site-packages:$PYTHONPATH \
  python3 tools/openpi_latency.py --port 8000 --n 40 --step-hz 10 --horizon 50 --tag sim

# 3. One-shot pipeline sanity check. Freezes one live observation and interrogates
#    the server: is the client→server contract intact, or is the policy simply not
#    responding to what it is shown? Run with cameras up but the client node DOWN.
PYTHONPATH=~/Franka/.venv/lib/python3.12/site-packages:$PYTHONPATH \
  python3 tools/openpi_sanity.py --prompt "pick up the mustard bottle"

# 4. Replay a TRAINING frame through the live server. Settles "is the pipeline broken
#    or is it the sim-to-real visual gap?" — the server has seen these exact pixels,
#    so it must reproduce roughly the recorded actions. Also settles the front-camera
#    swap offline (run with and without --swap-front). Needs pyarrow.
~/Franka/.venv/bin/python tools/openpi_replay.py --episode 0 --frame 120 [--swap-front]

# 5. Live-updating plot of the debug .npy files while a run is in progress.
python3 franka_openpi/plot_debug_live.py --debug_dir ~/Franka/src/franka_openpi/debug
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No subscriber on /fer_arm_controller/joint_trajectory after 15s` | Controller not loaded/active on the Franka PC. | Check `ros2 control list_controllers`; verify `ROS_DOMAIN_ID` / DDS reach. |
| `Server reports rtc_active=False` | Server build doesn't implement RTC, or rejects the `prev_actions_*` keys. | Use the RTC-capable `serve_policy.py`; check key names and `prev_actions_start/d`. |
| Arm moves, then nothing, silently | Trajectory rejected because the final point carried non-zero velocity. | Don't touch `_waypoint_velocities`. Terminal velocity must be exactly 0. |
| Arm stalls mid-episode, log looks normal | Collision reflex; the arm ignores commands and `_chunk_index()` is wall-time-driven, so the loop keeps splicing into a stationary arm. | `auto_error_recovery:=true`. |
| Vibration / buzzing at faster cadence | Chunk zigzag → acceleration scales as 1/step_duration². | Raise `smooth_window` (15/4 is the tuned point). |
| One jerk per chunk handover | Server residual not absorbed, or a frozen-region violation. | Raise `splice_blend_steps`; raise `rtc_d`. |
| Lurch-then-crawl at every republish | Catch-up segment mis-timed. | Check `CATCHUP_MAX_SEC` is still absolute (0.6 s), and that `/joint_states` publishes `velocity` so `v_now` is real. |
| `parameter not set` on `joint_stiffness:="[]"` | An empty YAML list carries no element type, so launch_ros passes `PARAMETER_NOT_SET`. | Already handled — `_float_list_param` catches it and treats it as "leave the robot alone". |
| Compliance services time out | `franka_hardware` isn't up (mock hardware, or the Franka PC launch failed). | Warning only; the arm keeps whatever stiffness it had. Check the Franka PC. |
| `ModuleNotFoundError: openpi_client` | venv path wrong. | Fix `_VENV_SITE` at the top of the launch file (`~/Franka/.venv/lib/python3.12/site-packages`). |
| `ModuleNotFoundError: omni_place` | `franka-vision-guided-manipulation` not built. | `colcon build` it in the same workspace. |
| Websocket dies mid-inference | Default 20 s keepalive ping fires during a long inference. | Already handled — `_NoPingPolicy` disables pings on both client nodes. |

---

## Package layout

```
franka_openpi/
├── franka_openpi/
│   ├── openpi_rtc_node.py      ← RTC client: obs packing, d/s scheduling, smoothing,
│   │                             splice blending, diagnostics.  THE main file.
│   ├── rtc_executor.py         ← streaming executor: JTC topic publish, time
│   │                             parameterization, catch-up sizing, gripper scheduling
│   ├── openpi_client_node.py   ← blocking baseline client
│   ├── action_executor.py      ← blocking executor + shared limits/gripper/constants
│   ├── robot_compliance.py     ← stiffness, collision thresholds, reflex recovery
│   ├── camera_viewer.py        ← live view of the three streams
│   ├── plot_debug_live.py      ← watch debug/*.npy, regenerate the PNG
│   ├── groot_client_node.py    ← GR00T policy variant
│   ├── act_client_node.py      ← ACT policy variant
│   └── hand_eye_calibrate.py
├── launch/
│   ├── openpi_franka_RTC.launch_sim.py   ← sim checkpoint + full RTC knobs  ★
│   ├── openpi_franka_RTC.launch.py       ← real-data checkpoint, core RTC only
│   ├── openpi_franka.launch_sim.py       ← blocking baseline, sim checkpoint
│   ├── openpi_franka.launch.py           ← blocking baseline, real checkpoint
│   ├── camera_launcher.launch.py         ← cameras alone
│   ├── act_franka.launch.py / groot_franka.launch.py
├── tools/                      ← rtc_logic_check, openpi_latency, openpi_sanity, openpi_replay
└── debug/                      ← run artefacts (rtc/, latency/, sanity/)
```

`rtc_executor` composes an `ActionExecutor` purely for the gripper path, so both
pipelines drive the gripper identically; its `FollowJointTrajectory` client is created
but never used. Nothing in `action_executor.py` is modified by the RTC path — the limit
constants and gripper code are imported so the two stay in sync.
