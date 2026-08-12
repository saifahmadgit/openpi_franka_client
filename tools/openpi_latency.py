#!/usr/bin/env python3
"""Client-side round-trip latency probe for the OpenPI policy server.

Step 1 of the RTC work: measure `d`, the inference latency expressed in control
steps, before deciding whether real-time chunking is worth building.

What it measures is the wall clock the *client* blocks for -- pack + send +
network + server queue + inference + reply + unpack. That is deliberately not
the server's `policy_timing.infer_ms`, which excludes everything outside the
model call; when it is present in the reply we print both and report the
difference as transport overhead.

Nothing here touches the robot: no action clients are created, so this cannot
move the arm. It is safe to run while the arm is powered but idle.

    source /opt/ros/jazzy/setup.bash
    source ~/Franka/install/setup.bash
    PYTHONPATH=~/Franka/.venv/lib/python3.12/site-packages:$PYTHONPATH \
        python3 ~/Franka/src/franka_openpi/tools/openpi_latency.py \
        --port 8000 --n 40 --prompt "pick up the cracker box"

With cameras up it sends live frames (the faithful measurement). Without them,
`--synthetic` sends correctly-shaped noise; the payload size and therefore the
serialization + wire cost are identical, only the image content differs.
"""
import argparse
import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import websockets.sync.client

# Same import path the node uses; needs the venv on PYTHONPATH.
from openpi_client import msgpack_numpy, websocket_client_policy

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

OUT_DIR = os.path.expanduser("~/Franka/src/franka_openpi/debug/latency")


class _NoPingTimedPolicy(websocket_client_policy.WebsocketClientPolicy):
    """Keepalive pings disabled (as in openpi_client_node), plus a timing split.

    The split matters because the two halves scale differently: pack/unpack grow
    with the number of cameras, the wire+server term is what a faster GPU or a
    quieter network would move.
    """

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=None,
                    ping_timeout=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                print(f"  still waiting for server at {self._uri} ...")
                time.sleep(5)

    def infer_timed(self, obs: Dict) -> Tuple[Dict, Dict]:
        t0 = time.monotonic()
        data = self._packer.pack(obs)
        t1 = time.monotonic()
        self._ws.send(data)
        response = self._ws.recv()
        t2 = time.monotonic()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        result = msgpack_numpy.unpackb(response)
        t3 = time.monotonic()
        return result, {
            "total_ms": (t3 - t0) * 1e3,
            "pack_ms": (t1 - t0) * 1e3,
            "wire_server_ms": (t2 - t1) * 1e3,
            "unpack_ms": (t3 - t2) * 1e3,
            "bytes_sent": len(data),
            "bytes_recv": len(response),
        }


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


def synthetic_frames(two_front):
    """CHW uint8 noise at the RealSense color profile the launch files set."""
    rng = np.random.default_rng(0)
    keys = ["front_1", "wrist"] + (["front_2"] if two_front else [])
    return {
        k: np.ascontiguousarray(rng.integers(0, 256, (3, 480, 640), dtype=np.uint8))
        for k in keys
    }


def live_frames(two_front, timeout_s=15.0):
    """One frame per camera plus the latest joint state, straight off the topics."""
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState

    class Grabber(Node):
        def __init__(self):
            super().__init__("openpi_latency")
            self.bridge = CvBridge()
            self.images = {k: None for k in TOPICS}
            self.joints = None
            for key, topic in TOPICS.items():
                self.create_subscription(
                    Image, topic, lambda m, k=key: self._img_cb(m, k), 10
                )
            self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        def _img_cb(self, msg, key):
            img = self.bridge.imgmsg_to_cv2(msg, "rgb8")
            self.images[key] = np.ascontiguousarray(img.transpose(2, 0, 1))

        def _joint_cb(self, msg):
            q = np.zeros(9) if self.joints is None else self.joints
            name_to_pos = dict(zip(msg.name, msg.position))
            for i, jname in enumerate(JOINT_ORDER):
                if jname in name_to_pos:
                    q[i] = name_to_pos[jname]
            self.joints = q

        def ready(self):
            if self.joints is None or self.images["wrist"] is None:
                return False
            if self.images["front_1"] is None:
                return False
            return not (two_front and self.images["front_2"] is None)

    rclpy.init()
    node = Grabber()
    deadline = time.time() + timeout_s
    while not node.ready() and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    ok = node.ready()
    images = {k: v.copy() for k, v in node.images.items() if v is not None}
    state = node.joints.copy() if node.joints is not None else None
    node.destroy_node()
    rclpy.shutdown()
    if not ok:
        missing = [k for k, v in node.images.items() if v is None]
        raise RuntimeError(
            f"cameras/joints not ready after {timeout_s:.0f}s (missing images={missing}, "
            f"joints={'yes' if state is not None else 'no'}). Re-run with --synthetic "
            f"to measure transport + inference without the robot."
        )
    return images, state


def pct(a, q):
    return float(np.percentile(a, q))


def report(total, args, label, rows=None, srv=None):
    dt_ms = 1000.0 / args.step_hz
    print("\n" + "=" * 72)
    print(f"ROUND TRIP, measured at the client — {label} ({len(total)} calls)")
    print("=" * 72)
    print(f"  min   {total.min():8.1f} ms")
    print(f"  p50   {pct(total, 50):8.1f} ms")
    print(f"  p95   {pct(total, 95):8.1f} ms")
    print(f"  p99   {pct(total, 99):8.1f} ms")
    print(f"  max   {total.max():8.1f} ms")
    print(f"  mean  {total.mean():8.1f} ms   std {total.std():.1f}")
    if rows:
        print(f"\n  breakdown (mean): pack {np.mean([r['pack_ms'] for r in rows]):.1f} ms, "
              f"wire+server {np.mean([r['wire_server_ms'] for r in rows]):.1f} ms, "
              f"unpack {np.mean([r['unpack_ms'] for r in rows]):.1f} ms")
        print(f"  payload: {rows[0]['bytes_sent'] / 1e6:.2f} MB up, "
              f"{rows[0]['bytes_recv'] / 1e3:.1f} kB down")
    if srv is not None and np.isfinite(srv).all():
        print(f"  server infer_ms: p50 {pct(srv, 50):.1f}, p95 {pct(srv, 95):.1f} — "
              f"transport+serialization is the remaining "
              f"{pct(total, 50) - pct(srv, 50):.1f} ms at p50")

    print("\n" + "=" * 72)
    print(f"d IN CONTROL STEPS   (step = {dt_ms:.0f} ms @ {args.step_hz:g} Hz, H = {args.horizon})")
    print("=" * 72)
    for lbl, v in (("p50", pct(total, 50)), ("p95", pct(total, 95)), ("max", float(total.max()))):
        d = v / dt_ms
        print(f"  {lbl}: d = {d:5.2f} steps  ->  d/H = {d / args.horizon:.3f}"
              f"   (re-query at index {max(0, args.horizon - int(np.ceil(d)))})")
    print("\n  RTC sizes d off the tail: one slow call is what produces a visible jerk,")
    print("  so the p95/max row is the one to design against, not p50.")


def from_spans(args):
    """Re-derive the latency table from an existing run's query_spans.npy.

    openpi_client_node already logs (t_request, t_received) around every infer()
    call, so past episodes carry the measurement without re-running anything —
    and without editing the node that produced them. The first span includes JIT
    compilation and is dropped unless --keep-first.
    """
    spans = np.load(os.path.expanduser(args.from_spans))
    total = (spans[:, 1] - spans[:, 0]) * 1e3
    print(f"Loaded {len(total)} spans from {args.from_spans}")
    if len(total) > 1 and not args.keep_first:
        print(f"  dropping first span ({total[0] / 1e3:.1f}s — model warm-up/JIT)")
        total = total[1:]
    gaps = (spans[1:, 0] - spans[:-1, 1])
    if len(gaps):
        print(f"  execution windows between calls: p50 {pct(gaps, 50):.2f}s, "
              f"min {gaps.min():.2f}s  (chunk playback time — the budget d has to fit in)")
        chunk_path = args.from_spans.replace("query_spans.npy", "chunks.npy")
        if os.path.exists(os.path.expanduser(chunk_path)):
            sizes = np.load(os.path.expanduser(chunk_path))[:len(gaps)]
            sec_per_step = gaps / np.maximum(sizes, 1)
            print(f"  chunk sizes {sizes.tolist()} -> {sec_per_step.mean():.3f} s/step "
                  f"= {1 / sec_per_step.mean():.2f} Hz effective control rate")
            print(f"  (nominal STEP_DURATION is {1 / args.step_hz:.3f} s = {args.step_hz:g} Hz; "
                  f"the gap is _time_parameterize stretching segments to the {'':s}"
                  f"velocity/accel limits, plus the synchronous plotting in _save_logs)")
    report(total, args, os.path.basename(args.from_spans))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="129.105.69.11")
    ap.add_argument("--port", type=int, default=8000,
                    help="serve_policy.py --port (the launch files currently use 8001)")
    ap.add_argument("--prompt", default="pick up the cracker box")
    ap.add_argument("--two-front-cameras", default="true")
    ap.add_argument("--n", type=int, default=40, help="timed calls after warm-up")
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed calls first; the very first call JIT-compiles and can "
                         "take tens of seconds, which would swamp the tail statistics")
    ap.add_argument("--step-hz", type=float, default=5.0,
                    help="control rate used to convert latency into steps of the chunk "
                         "(action_executor.STEP_DURATION is currently 1/5 s)")
    ap.add_argument("--horizon", type=int, default=50, help="action_horizon H")
    ap.add_argument("--synthetic", action="store_true",
                    help="send shaped noise instead of live camera frames")
    ap.add_argument("--refresh-obs", action="store_true",
                    help="re-grab live frames before every call (adds ROS grab cost; off "
                         "by default so the number isolates the server round trip)")
    ap.add_argument("--tag", default="", help="output filename prefix, e.g. sim")
    ap.add_argument("--from-spans", default="",
                    help="skip the live probe; recompute the table from a past run's "
                         "debug/<tag>_query_spans.npy instead")
    ap.add_argument("--keep-first", action="store_true",
                    help="--from-spans: keep the warm-up span in the statistics")
    args = ap.parse_args()
    two_front = args.two_front_cameras.lower() in ("1", "true", "yes")
    prefix = f"{args.tag}_" if args.tag else ""

    if args.from_spans:
        from_spans(args)
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.synthetic:
        print("Observation source: SYNTHETIC noise (payload size identical to live frames)")
        images = synthetic_frames(two_front)
        state = np.zeros(9, dtype=np.float32)
    else:
        print("Observation source: LIVE camera topics + /joint_states")
        images, state = live_frames(two_front)

    obs = build_obs(images, state, args.prompt, two_front)

    print(f"Connecting to ws://{args.host}:{args.port} ...")
    policy = _NoPingTimedPolicy(host=args.host, port=args.port)
    print(f"  server metadata: {policy.get_server_metadata()}")

    print(f"\nWarm-up ({args.warmup} calls, untimed — the first one compiles the model)")
    for i in range(args.warmup):
        t = time.monotonic()
        result, _ = policy.infer_timed(obs)
        print(f"  warmup {i + 1}: {(time.monotonic() - t):.2f}s  "
              f"actions {np.asarray(result['actions']).shape}")

    print(f"\nTiming {args.n} calls ...")
    rows = []
    for i in range(args.n):
        if args.refresh_obs and not args.synthetic:
            images, state = live_frames(two_front)
            obs = build_obs(images, state, args.prompt, two_front)
        result, t = policy.infer_timed(obs)
        srv = result.get("policy_timing", {}).get("infer_ms")
        t["server_infer_ms"] = float(srv) if srv is not None else float("nan")
        rows.append(t)
        print(f"  [{i + 1:3d}/{args.n}] total {t['total_ms']:7.1f} ms   "
              f"pack {t['pack_ms']:5.1f}   wire+server {t['wire_server_ms']:7.1f}   "
              f"unpack {t['unpack_ms']:5.1f}"
              + (f"   server_infer {t['server_infer_ms']:7.1f}" if srv is not None else ""))

    total = np.array([r["total_ms"] for r in rows])
    srv = np.array([r["server_infer_ms"] for r in rows])
    have_srv = bool(np.isfinite(srv).all())
    if not have_srv:
        print("\n  (server did not report policy_timing.infer_ms)")

    report(total, args, "live probe", rows=rows, srv=srv if have_srv else None)

    out = os.path.join(OUT_DIR, prefix + "latency_ms.npy")
    np.save(out, total)
    meta = {
        "host": args.host, "port": args.port, "n": args.n, "warmup": args.warmup,
        "two_front_cameras": two_front, "synthetic": args.synthetic,
        "refresh_obs": args.refresh_obs, "step_hz": args.step_hz, "horizon": args.horizon,
        "prompt": args.prompt,
        "total_ms": {"min": float(total.min()), "p50": pct(total, 50),
                     "p95": pct(total, 95), "p99": pct(total, 99),
                     "max": float(total.max()), "mean": float(total.mean()),
                     "std": float(total.std())},
        "server_infer_ms": ({"p50": pct(srv, 50), "p95": pct(srv, 95)} if have_srv else None),
        "bytes_sent": rows[0]["bytes_sent"], "bytes_recv": rows[0]["bytes_recv"],
    }
    with open(os.path.join(OUT_DIR, prefix + "latency.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  saved -> {out}")
    print(f"  saved -> {os.path.join(OUT_DIR, prefix + 'latency.json')}")


if __name__ == "__main__":
    main()
