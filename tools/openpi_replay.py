#!/usr/bin/env python3
"""Replay a TRAINING frame through the live policy server.

The deploy-time question "is the pipeline broken or is it the visual gap?" cannot be
answered from real-robot frames, because a wrong answer there is consistent with both.
Training frames settle it: the server has seen these exact pixels 2130 episodes' worth,
so it must reproduce roughly the recorded actions. If it does, every link from
observation construction through msgpack transport to action decoding is clean and the
only remaining explanation for real-world failure is the visual gap. If it does not,
there is a contract bug and no amount of domain randomization will fix it.

It also settles the front-camera swap offline. Run with and without --swap-front and
keep whichever mapping predicts the recorded actions better; that is the mapping
training used, no robot required.

    /home/mdsaifahmad/Franka/.venv/bin/python \
        ~/Franka/src/franka_openpi/tools/openpi_replay.py --episode 0 --frame 120

Needs pyarrow for the parquet state column:
    /home/mdsaifahmad/Franka/.venv/bin/python -m pip install pyarrow
No ROS, no robot, nothing is commanded.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/openpi-client/src"))
from openpi_client import websocket_client_policy  # noqa: E402

REPO = "saifahmad123/Franka_GraspNet_Test"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
CACHE = os.path.expanduser("~/Franka/src/franka_openpi/debug/replay_cache")
CAMS = ["front_1", "front_2", "wrist"]


def fetch(rel_path: str) -> str:
    """Download <BASE>/<rel_path> into the cache once, return the local path."""
    local = os.path.join(CACHE, rel_path.replace("/", "_"))
    os.makedirs(CACHE, exist_ok=True)
    if not os.path.exists(local) or os.path.getsize(local) == 0:
        print(f"  downloading {rel_path} ...")
        subprocess.run(
            ["curl", "-sL", "--fail", "--max-time", "300", f"{BASE}/{rel_path}", "-o", local],
            check=True,
        )
    return local


def load_episode(ep: int):
    """Return (states (T,9), actions (T,8), {cam: local mp4 path}) for one episode."""
    chunk = ep // 1000
    parquet = fetch(f"data/chunk-{chunk:03d}/episode_{ep:06d}.parquet")
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow missing — install it into the venv (see docstring)")
    tbl = pq.read_table(parquet)
    states = np.stack(tbl["observation.state"].to_pylist()).astype(np.float32)
    actions = np.stack(tbl["action"].to_pylist()).astype(np.float32)
    videos = {
        c: fetch(f"videos/chunk-{chunk:03d}/observation.images.{c}/episode_{ep:06d}.mp4")
        for c in CAMS
    }
    return states, actions, videos


def read_frame(path: str, idx: int) -> np.ndarray:
    """Frame `idx` as CHW uint8 RGB — byte-identical to _proc_image in the client node."""
    import cv2

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx} of {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1))


def build_obs(imgs, state, prompt, two_front, swap_front):
    """Exactly openpi_client_node._get_observation, with an optional front swap."""
    f1, f2 = imgs["front_1"], imgs["front_2"]
    if swap_front:
        f1, f2 = f2, f1
    if two_front:
        cams = {"cam_high": imgs["wrist"], "cam_left_wrist": f1, "cam_right_wrist": f2}
    else:
        cams = {"cam_high": f1, "cam_left_wrist": imgs["wrist"]}
    return {"state": state[:9].astype(np.float32), "images": cams, "prompt": prompt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="129.105.69.11")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--prompt", default="pick up the cracker box")
    ap.add_argument("--two-front-cameras", default="true")
    ap.add_argument("--swap-front", action="store_true",
                    help="exchange front_1/front_2 — the mapping openpi_franka.launch_sim.py uses")
    args = ap.parse_args()
    two_front = args.two_front_cameras.lower() in ("1", "true", "yes")

    print(f"Loading training episode {args.episode} ...")
    states, actions, videos = load_episode(args.episode)
    T = len(states)
    if not 0 <= args.frame < T:
        sys.exit(f"--frame must be in [0,{T}) for this episode")
    print(f"  {T} frames, replaying frame {args.frame}")

    imgs = {c: read_frame(p, args.frame) for c, p in videos.items()}
    state = states[args.frame]

    policy = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    np.set_printoptions(precision=3, suppress=True)

    def query(swap):
        obs = build_obs(imgs, state, args.prompt, two_front, swap)
        return np.array(policy.infer(obs)["actions"])[:, :8]

    print("\nQuerying (first call includes warm-up, can take ~2 min) ...")
    pred = query(args.swap_front)
    H = min(len(pred), T - args.frame)
    truth = actions[args.frame:args.frame + H]
    pred = pred[:H]

    # Error against the recorded continuation. Meaningless without a scale, so it is
    # compared below against the error of an unrelated frame's actions.
    mae = np.abs(pred[:, :7] - truth[:, :7]).mean()

    rng = np.random.default_rng(0)
    far = (args.frame + T // 2) % max(T - H, 1)
    wrong = actions[far:far + H][:, :7]
    n = min(len(wrong), H)
    mae_wrong = np.abs(truth[:n, :7] - wrong[:n]).mean()

    print("\n" + "=" * 68)
    print("REPLAY RESULT")
    print("=" * 68)
    print(f"  front mapping         : {'SWAPPED (launch_sim)' if args.swap_front else 'normal (launch.py)'}")
    print(f"  predicted vs recorded : MAE {mae:.4f} rad over {H} steps, 7 arm joints")
    print(f"  unrelated-frame base  : MAE {mae_wrong:.4f} rad  (what 'wrong' looks like)")
    print(f"  per-joint MAE         : {np.abs(pred[:, :7] - truth[:, :7]).mean(0)}")
    print(f"  gripper pred [{pred[:, 7].min():.3f},{pred[:, 7].max():.3f}]  "
          f"recorded [{truth[:, 7].min():.3f},{truth[:, 7].max():.3f}]")

    print("\n  reading:")
    if mae < 0.25 * mae_wrong:
        print("  >> PIPELINE CLEAN. The server reproduces training behaviour through your")
        print("     exact client path, so observation construction, transport, normalization")
        print("     and action decoding are all correct. Real-world failure is then the")
        print("     visual gap, and only training-side fixes will move it.")
    elif mae < 0.6 * mae_wrong:
        print("  >> PARTIAL. Better than chance but not tracking the recording. Suspect the")
        print("     camera mapping first — rerun with the opposite --swap-front setting.")
    else:
        print("  >> BROKEN. The policy cannot reproduce its own training data through this")
        print("     client. This is a contract bug, not a domain gap. Compare the camera")
        print("     key mapping, the state layout and the prompt against the training config.")
    print("\n  Run again with the opposite --swap-front to settle the front-camera question:")
    print("  the mapping with the LOWER MAE is the one training used.")


if __name__ == "__main__":
    main()
