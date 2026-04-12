# franka_openpi

A ROS 2 package that runs a [pi0.5 / OpenPI](https://github.com/Physical-Intelligence/openpi) policy on a **Franka Panda** robot.  
The node streams camera images and joint states to a remote GPU server running the OpenPI inference server, receives end-effector action chunks in return, and executes them via MoveIt 2 using the companion package below.

**Depends on:** [franka-vision-guided-manipulation](https://github.com/saifahmadgit/franka-vision-guided-manipulation) for the MoveIt 2 motion-planning interface (`MotionPlanningInterface` / `omni_place`).

---

## System overview

```
Robot PC                                GPU Server
───────────────────────────────         ──────────────────────────────
 RealSense front  ──┐
 RealSense wrist  ──┤  openpi_client_node         openpi serve
 /joint_states    ──┘   │  packs obs dict    ──►  (pi0.5 model)
                         │  [cam_high, cam_left_wrist, state, prompt]
                         │◄─────────────────────  actions (N, 8)
                         │  [x, y, z, qw, qx, qy, qz, gripper]
                         ▼
               ActionExecutor
               (MotionPlanningInterface)
               MoveIt 2 → Franka hardware
```

---

## Prerequisites

### Robot PC

| Requirement | Notes |
|---|---|
| Ubuntu 22.04 | |
| ROS 2 Humble | full desktop install |
| [franka-vision-guided-manipulation](https://github.com/saifahmadgit/franka-vision-guided-manipulation) | must be built in the same workspace — provides `omni_place` |
| `realsense2_camera` ROS package | `sudo apt install ros-humble-realsense2-camera` |
| `cv_bridge`, `numpy`, `opencv-python` | |
| `openpi_client` Python package | installed in a `.venv` (see below) |

### GPU Server

Follow the [OpenPI server setup](https://github.com/Physical-Intelligence/openpi) to serve a pi0.5 checkpoint:

```bash
# on the GPU server
python scripts/serve_policy.py --env FRANKA_EXAMPLE --host 0.0.0.0 --port 8000
```

---

## Installation

```bash
# 1. Clone into your ROS 2 workspace src/
cd ~/ros2_ws/src
git clone https://github.com/saifahmadgit/openpi_franka_client franka_openpi

# 2. Also clone the MoveIt interface dependency (if not already present)
git clone https://github.com/saifahmadgit/franka-vision-guided-manipulation

# 3. Install openpi_client into a venv at the workspace root
#    (the launch file automatically prepends this venv to PYTHONPATH)
cd ~/ros2_ws
python3 -m venv .venv
source .venv/bin/activate
pip install openpi-client        # or follow OpenPI's own install guide

# 4. Build the workspace
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Hardware setup

Two Intel RealSense cameras are expected:

| Role | Namespace | Serial (example) | Resolution |
|---|---|---|---|
| Front / overhead | `/camera/front` | `938422076824` | 640×480 @ 30 fps |
| Wrist-mounted | `/camera/wrist` | `233522075872` | 640×480 @ 30 fps |

Update the serial numbers in `launch/openpi_franka.launch.py` (or pass them as launch arguments) to match your cameras.

---

## Configuration

All parameters are set in `launch/openpi_franka.launch.py`:

| Parameter | Default | Description |
|---|---|---|
| `server_host` | `129.105.69.11` | IP of the GPU machine running `openpi serve` |
| `server_port` | `8000` | WebSocket port |
| `prompt` | `pick up the red apple` | Language instruction sent with every observation |
| `action_horizon` | `10` | Number of waypoints per inference call |
| `num_episodes` | `5` | How many full episodes to run |

---

## Running

### 1. Start the OpenPI inference server (GPU machine)

Clone and set up [openpi_franka](https://github.com/saifahmadgit/openpi_franka) on the GPU machine, then serve the policy:

```bash
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/serve_policy.py \
  --default_prompt "pick up the apple" \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_magicsim_apple_red \
  --policy.dir checkpoints/pi05_magicsim_apple_red/magicsim_apple_lora_red/55000
```

- `CUDA_VISIBLE_DEVICES=1` — use GPU index 1 (adjust if needed)
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` — prevents JAX from grabbing all GPU memory upfront
- `--policy.dir` — path to the checkpoint directory relative to the repo root

### 2. Launch MoveIt 2 on the Franka computer

On the robot PC, using the [franka-vision-guided-manipulation](https://github.com/saifahmadgit/franka-vision-guided-manipulation) workspace:

```bash
source ~/ros2_ws/install/setup.bash
ros2 launch omni_place omni_place.launch.py robot_ip:=<ROBOT_IP>
```

This brings up the Franka driver, MoveIt 2, and the `MotionPlanningInterface` that `ActionExecutor` calls into.

### 3. Launch cameras + OpenPI client (Robot PC)

```bash
# new terminal — cameras + OpenPI client
source ~/ros2_ws/install/setup.bash
ros2 launch franka_openpi openpi_franka.launch.py
```

To override parameters at launch time:

```bash
ros2 launch franka_openpi openpi_franka.launch.py \
  server_host:=192.168.1.50 \
  prompt:="stack the blue block on the red block" \
  num_episodes:=3
```

---

## Observation format sent to the server

```python
{
    "state":  np.ndarray,   # shape (9,)  float32 — [joint1-7, finger1, finger2]
    "images": {
        "cam_high":       np.ndarray,  # (3, 512, 512) CHW uint8  — front camera
        "cam_left_wrist": np.ndarray,  # (3, 256, 256) CHW uint8  — wrist camera
    },
    "prompt": str,          # language instruction
}
```

Images are center-cropped from 640×480 to 480×480 before resizing.

## Action format received from the server

```
actions: (action_horizon, 8)
  columns: [x, y, z, qw, qx, qy, qz, gripper]
  frame:   world frame (z = 0 at floor)
```

`ActionExecutor` subtracts 1.0 m from z to convert to the robot base frame, then sends the **final waypoint** of each chunk to MoveIt 2 as a Cartesian goal. Gripper threshold: > 0.5 → close, ≤ 0.5 → open.

---

## Package structure

```
franka_openpi/
├── franka_openpi/
│   ├── openpi_client_node.py   # ROS 2 node — observation packing + inference loop
│   └── action_executor.py      # sends EEF poses + gripper cmds to MoveIt 2
├── launch/
│   ├── openpi_franka.launch.py # Python launch file (cameras + client node)
│   └── openpi_franka.launch.xml
├── package.xml
└── setup.py
```

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Cameras not ready after 10 s` | Check `ros2 topic list` for `/camera/front/camera/color/image_raw` and `/camera/wrist/camera/color/image_raw`; verify serial numbers |
| `ImportError: No module named 'openpi_client'` | The `.venv` path in `launch/openpi_franka.launch.py` must match your workspace root; rebuild after moving the venv |
| Actions look reasonable but robot doesn't move | Confirm MoveIt 2 is running and `MotionPlanningInterface` from `franka-vision-guided-manipulation` initialised correctly |
| z coordinate way off | Check `WORLD_TO_BASE_Z = 1.0` in `action_executor.py` — adjust to your actual table/floor height |

---

## License

Apache 2.0
