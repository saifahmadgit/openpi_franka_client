#!/usr/bin/env python3
"""One-shot sanity check for the OpenPI deployment pipeline.

Freezes a single live observation (built exactly the way openpi_client_node builds
it), then interrogates the server with it. Answers one question: is the client→server
contract intact, or is the policy simply not responding to what it is being shown?

Run it with the robot + cameras up, but WITHOUT openpi_client_node running (only one
process should hold the /joint_states + camera subscriptions at inference time; two
clients on the same server is also fine but confuses the timing numbers).

    source /opt/ros/jazzy/setup.bash
    source ~/Franka/install/setup.bash
    PYTHONPATH=~/Franka/.venv/lib/python3.12/site-packages:$PYTHONPATH \
        python3 ~/Franka/src/franka_openpi/tools/openpi_sanity.py \
        --prompt "pick up the cracker box"

Nothing is sent to the robot — no action client is created, so this cannot move
the arm.
"""
import argparse
import os
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

# Same import path the node uses; needs the venv on PYTHONPATH.
from openpi_client import websocket_client_policy

JOINT_ORDER = [
    "fer_joint1", "fer_joint2", "fer_joint3", "fer_joint4",
    "fer_joint5", "fer_joint6", "fer_joint7",
    "fer_finger_joint1", "fer_finger_joint2",
]

TOPICS = {
    "front_1": "/camera/front_1/camera/color/image_raw",
    "front_2": "/camera/front_2/camera/color/image_raw",
    "wrist": "/camera/wrist/camera/color/image_raw",
}

OUT_DIR = os.path.expanduser("~/Franka/src/franka_openpi/debug/sanity")


class Grabber(Node):
    """Collects one frame per camera plus the latest joint state, then stops."""

    def __init__(self):
        super().__init__("openpi_sanity")
        self.bridge = CvBridge()
        self.images = {k: None for k in TOPICS}
        self.joints = None                 # None until /joint_states arrives
        self.joint_names_seen = set()
        for key, topic in TOPICS.items():
            self.create_subscription(
                Image, topic, lambda m, k=key: self._img_cb(m, k), 10
            )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

    def _img_cb(self, msg, key):
        # rgb8 → CHW, byte-identical to openpi_client_node._proc_image.
        img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.images[key] = np.ascontiguousarray(img.transpose(2, 0, 1))

    def _joint_cb(self, msg):
        self.joint_names_seen.update(msg.name)
        q = np.zeros(9) if self.joints is None else self.joints
        name_to_pos = dict(zip(msg.name, msg.position))
        for i, jname in enumerate(JOINT_ORDER):
            if jname in name_to_pos:
                q[i] = name_to_pos[jname]
        self.joints = q

    def ready(self, two_front):
        if self.joints is None:
            return False
        if self.images["wrist"] is None or self.images["front_1"] is None:
            return False
        return not (two_front and self.images["front_2"] is None)


def build_obs(images_chw, state, prompt, two_front):
    """The exact key mapping openpi_client_node._get_observation uses."""
    if two_front:
        cams = {
            "cam_high": images_chw["wrist"],
            "cam_left_wrist": images_chw["front_1"],
            "cam_right_wrist": images_chw["front_2"],
        }
    else:
        cams = {
            "cam_high": images_chw["front_1"],
            "cam_left_wrist": images_chw["wrist"],
        }
    return {"state": state[:9].astype(np.float32), "images": cams, "prompt": prompt}


def save_montage(images_chw, two_front, path):
    """Save what the server sees, tiled and labelled by SERVER KEY.

    Labelling by server key (not by camera name) is the point: this is the image
    to hold next to a training frame to confirm the mapping is right.
    """
    import cv2

    keys = (
        [("cam_high", "wrist"), ("cam_left_wrist", "front_1"), ("cam_right_wrist", "front_2")]
        if two_front
        else [("cam_high", "front_1"), ("cam_left_wrist", "wrist")]
    )
    panels = []
    for server_key, cam in keys:
        chw = images_chw[cam]
        bgr = cv2.cvtColor(chw.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (426, 320))
        cv2.rectangle(bgr, (0, 0), (426, 44), (0, 0, 0), -1)
        cv2.putText(bgr, server_key, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bgr, f"<- {cam}", (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 200), 1, cv2.LINE_AA)
        panels.append(bgr)
    cv2.imwrite(path, np.hstack(panels))


def chunk_stats(a):
    return (
        f"shape={a.shape}  "
        f"per-joint travel(rad)={np.round(np.ptp(a[:, :7], axis=0), 3)}  "
        f"grip[{a[:, 7].min():.3f},{a[:, 7].max():.3f}]"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="129.105.69.11")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="pick up the cracker box")
    ap.add_argument("--two-front-cameras", default="true")
    ap.add_argument("--n-samples", type=int, default=8,
                    help="repeated draws on one frozen observation (mode-spread probe)")
    args = ap.parse_args()
    two_front = args.two_front_cameras.lower() in ("1", "true", "yes")

    os.makedirs(OUT_DIR, exist_ok=True)
    rclpy.init()
    node = Grabber()

    print("Waiting for cameras + /joint_states ...")
    deadline = time.time() + 15.0
    while not node.ready(two_front) and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.ready(two_front):
        missing = [k for k, v in node.images.items() if v is None]
        print(f"FAIL: not ready. missing images={missing} joints={node.joints is not None}")
        return

    state = node.joints.copy()
    images = {k: v.copy() for k, v in node.images.items() if v is not None}
    node.destroy_node()
    rclpy.shutdown()

    # ── 1. state vector ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("1. STATE (the 9 numbers the policy is normalized against)")
    print("=" * 72)
    np.set_printoptions(precision=4, suppress=True)
    for i, name in enumerate(JOINT_ORDER):
        print(f"  [{i}] {name:<20} {state[i]:+.4f}")
    missing_joints = [j for j in JOINT_ORDER if j not in node.joint_names_seen]
    if missing_joints:
        print(f"\n  *** {len(missing_joints)} joint(s) NEVER appeared on /joint_states: "
              f"{missing_joints}")
        print("      Those slots are hard 0.0 in every observation ever sent. If the")
        print("      fingers are among them the gripper channel is dead and the state")
        print("      is mis-normalized against the training stats.")
    else:
        print("\n  all 9 joints present on /joint_states  OK")

    # ── 2. what the server actually sees ──────────────────────────────────────
    montage = os.path.join(OUT_DIR, "obs_as_sent.png")
    save_montage(images, two_front, montage)
    print("\n" + "=" * 72)
    print("2. IMAGES")
    print("=" * 72)
    print(f"  saved -> {montage}")
    print("  Open it next to a frame from the SIM training episodes. Each panel is")
    print("  labelled with the server key it fills. Same viewpoint per key, or the")
    print("  mapping/serials are wrong.")
    for cam, chw in images.items():
        print(f"  {cam:<8} dtype={chw.dtype} shape={chw.shape} mean={chw.mean():6.1f}")

    # ── 3. baseline query ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("3. BASELINE QUERY  (first call includes model warm-up, can take ~2 min)")
    print("=" * 72)
    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    meta = policy.get_server_metadata()
    print(f"  server metadata: {meta}")

    t = time.time()
    raw = np.array(policy.infer(build_obs(images, state, args.prompt, two_front))["actions"])
    print(f"  latency {time.time() - t:.1f}s   raw action shape {raw.shape}")
    if raw.shape[1] > 8:
        tail = raw[:, 8:]
        print(f"  columns 8..{raw.shape[1]-1} (assumed padding): "
              f"absmax={np.abs(tail).max():.4f}")
        if np.abs(tail).max() > 1e-3:
            print("  *** those columns are NOT zero — the [:, :8] slice may be dropping"
                  " real output, check the action space of the checkpoint")
    base = raw[:, :8]
    print(f"  {chunk_stats(base)}")
    d0 = np.abs(base[0, :7] - state[:7]).max()
    print(f"  |action[0] - current q|max = {d0:.4f} rad")
    if d0 > 0.3:
        print("  *** first action is far from where the arm actually is — that is an")
        print("      absolute-vs-delta action-space mismatch, not a policy quality issue")
    np.save(os.path.join(OUT_DIR, "baseline_chunk.npy"), base)

    # ── 4. ablations on the SAME frozen observation ───────────────────────────
    # Everything below reuses one static observation, so any difference in the
    # returned chunk is caused only by the field that was perturbed.
    print("\n" + "=" * 72)
    print("4. ABLATIONS — how much does the output move when an input is destroyed?")
    print("=" * 72)
    print("  (L2 = mean per-step distance from the baseline chunk, rad)")

    def probe(label, obs):
        a = np.array(policy.infer(obs)["actions"])[:, :8]
        n = min(len(a), len(base))
        l2 = float(np.linalg.norm(a[:n, :7] - base[:n, :7], axis=1).mean())
        print(f"  {label:<34} L2={l2:7.4f}   {chunk_stats(a)}")
        return l2

    black = {k: np.zeros_like(v) for k, v in images.items()}

    l2_same = probe("identical repeat (stochasticity)",
                    build_obs(images, state, args.prompt, two_front))
    l2_prompt = probe("nonsense prompt",
                      build_obs(images, state, "zzzz qqqq", two_front))
    l2_allblack = probe("ALL cameras black",
                        build_obs(black, state, args.prompt, two_front))

    per_cam = {}
    for cam in images:
        one = dict(images)
        one[cam] = np.zeros_like(images[cam])
        per_cam[cam] = probe(f"{cam} black", build_obs(one, state, args.prompt, two_front))

    jitter = state.copy()
    jitter[:7] += 0.15
    l2_state = probe("state +0.15 rad on every joint",
                     build_obs(images, jitter, args.prompt, two_front))

    # ── 4b. sampling spread on ONE frozen observation ─────────────────────────
    # pi0.5's action expert samples a mode rather than averaging, so the sharp
    # question is not "does the output change when I break an input" but "how
    # confident is the policy given this input". Informative conditioning
    # collapses the samples onto one trajectory; uninformative conditioning
    # leaves them scattered across whatever modes the prior contains.
    print("\n" + "-" * 72)
    print("  sampling spread — same observation queried N times")
    print("-" * 72)
    obs_fixed = build_obs(images, state, args.prompt, two_front)
    draws = [np.array(policy.infer(obs_fixed)["actions"])[:, :8] for _ in range(args.n_samples)]
    m = min(len(d) for d in draws)
    stack = np.stack([d[:m, :7] for d in draws])          # (N, steps, 7)
    spread = float(np.linalg.norm(stack.std(axis=0), axis=1).mean())
    endpoints = stack[:, -1, :]                            # (N, 7) final pose per draw
    end_spread = float(np.linalg.norm(endpoints - endpoints.mean(0), axis=1).mean())
    print(f"  {args.n_samples} draws, {m} steps each")
    print(f"  mean per-step std across draws : {spread:.4f} rad")
    print(f"  final-pose scatter             : {end_spread:.4f} rad")
    print("  (run this again on a TRAINING frame via openpi_replay.py for the")
    print("   comparison that matters — tight there and scattered here means the")
    print("   real images are contributing nothing.)")

    # ── 5. verdict ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("5. READING THE RESULT")
    print("=" * 72)
    noise = max(l2_same, 1e-4)
    print(f"  sampling noise floor (identical repeat): {noise:.4f}")

    if l2_allblack < 3 * noise:
        print("  >> VISION IS NOT DRIVING THE OUTPUT. Blanking every camera changed the")
        print("     chunk no more than resampling does. The policy is running off state")
        print("     + prompt alone, which is exactly what a sim-trained checkpoint does")
        print("     when the real images land outside its training distribution.")
        print("     This is a domain-gap result, NOT a plumbing bug.")
    else:
        print(f"  >> vision does affect the output ({l2_allblack:.4f} vs noise {noise:.4f}),")
        print("     so images are reaching the model and being used. Look at the mapping")
        print("     and the visual gap, not at the transport.")
    for cam, v in per_cam.items():
        share = v / l2_allblack if l2_allblack > 1e-6 else 0.0
        print(f"     {cam:<8} contributes {share*100:5.1f}% of the total vision effect")
    print(f"  prompt sensitivity {l2_prompt:.4f} — near the noise floor means the language")
    print("     conditioning is not doing anything either.")
    print(f"  state sensitivity  {l2_state:.4f} — should be clearly above noise; if it is")
    print("     not, the checkpoint is ignoring proprioception too.")
    print(f"\n  artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
