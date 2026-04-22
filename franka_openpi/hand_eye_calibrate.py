#!/usr/bin/env python3
"""
Hand-Eye Calibration: Find T_base → camera using ArUco marker.

Setup  : eye-in-hand (camera on end-effector), ArUco marker at fixed unknown pose.
Method : Each robot pose gives   T_base_marker = T_base_ee · X · T_cam_marker
         where X = T_ee_camera is the unknown rigid transform.
         Since T_base_marker is constant, we minimise its variance over all recordings.

Usage:
    python3 hand_eye_calibrate.py [options]

    Move the robot to a new configuration, wait for ArUco to be visible, then press R.
    Aim for ≥5 recordings with varied orientations.  Press Q when done.

Options (edit defaults below or pass as --arg value):
    --base-frame       robot base frame         (default: fr3_link0)
    --ee-frame         end-effector frame        (default: fr3_hand)
    --camera-frame     camera optical frame      (default: camera_color_optical_frame)
    --marker-frame     aruco marker TF frame     (default: marker_0)
    --out              output YAML file          (default: hand_eye_result.yaml)
"""

import sys
import tty
import termios
import threading
import argparse
import yaml

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

import rclpy
from rclpy.node import Node
import tf2_ros
from tf2_ros import TransformException

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def tf_to_mat(t) -> np.ndarray:
    """TransformStamped → 4×4 homogeneous matrix."""
    tr = t.transform.translation
    ro = t.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([ro.x, ro.y, ro.z, ro.w]).as_matrix()
    T[:3, 3]  = [tr.x, tr.y, tr.z]
    return T


def mat_to_xyzquat(M: np.ndarray):
    t = M[:3, 3]
    q = Rotation.from_matrix(M[:3, :3]).as_quat()   # (x, y, z, w)
    return t, q


def params_to_mat(p: np.ndarray) -> np.ndarray:
    """6-vector [tx,ty,tz, rx,ry,rz] → 4×4 matrix."""
    M = np.eye(4)
    M[:3, :3] = Rotation.from_rotvec(p[3:]).as_matrix()
    M[:3, 3]  = p[:3]
    return M


def mat_to_params(M: np.ndarray) -> np.ndarray:
    return np.concatenate([M[:3, 3],
                           Rotation.from_matrix(M[:3, :3]).as_rotvec()])

# ──────────────────────────────────────────────────────────────
# Calibrator node
# ──────────────────────────────────────────────────────────────

class HandEyeCalibrator(Node):
    def __init__(self, args):
        super().__init__('hand_eye_calibrator')

        self.base_frame   = args.base_frame
        self.ee_frame     = args.ee_frame
        self.cam_frame    = args.camera_frame
        self.marker_frame = args.marker_frame
        self.out_file     = args.out

        self.tf_buffer   = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=5))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Each entry: (T_base_ee [4×4], T_cam_marker [4×4])
        self.measurements: list[tuple[np.ndarray, np.ndarray]] = []

        # Current best estimate of T_ee_camera (6-vector)
        self._x0 = np.zeros(6)

    # ── Recording ────────────────────────────────────────────

    def record(self) -> bool:
        """Snapshot current TF pair. Returns True on success."""
        try:
            t_be = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            t_cm = self.tf_buffer.lookup_transform(
                self.cam_frame, self.marker_frame,
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.5))
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return False

        T_base_ee  = tf_to_mat(t_be)
        T_cam_marker = tf_to_mat(t_cm)

        self.measurements.append((T_base_ee, T_cam_marker))
        n = len(self.measurements)
        t_vec = T_cam_marker[:3, 3]
        self.get_logger().info(
            f'[#{n}] Recorded  — marker in camera: '
            f'x={t_vec[0]:.3f}  y={t_vec[1]:.3f}  z={t_vec[2]:.3f} m'
        )
        return True

    # ── Optimisation ─────────────────────────────────────────

    def _residual(self, p: np.ndarray) -> float:
        """
        Cost: variance of T_base_marker across measurements.
        All T_base_marker = T_base_ee · X · T_cam_marker must be identical.
        """
        X = params_to_mat(p)
        poses = [T_be @ X @ T_cm for T_be, T_cm in self.measurements]

        # Translational spread
        pos  = np.array([m[:3, 3] for m in poses])
        cost = np.sum((pos - pos.mean(axis=0)) ** 2)

        # Rotational spread (geodesic distance to mean quaternion)
        qs = np.array([Rotation.from_matrix(m[:3, :3]).as_quat() for m in poses])
        # Consistent hemisphere
        for i in range(1, len(qs)):
            if np.dot(qs[i], qs[0]) < 0:
                qs[i] = -qs[i]
        q_mean = qs.mean(axis=0)
        q_mean /= np.linalg.norm(q_mean)
        cost += np.sum(1.0 - np.abs(qs @ q_mean))

        return cost

    def optimise(self) -> np.ndarray | None:
        if len(self.measurements) < 2:
            self.get_logger().info('Need at least 2 measurements – skipping optimisation.')
            return None

        result = minimize(
            self._residual, self._x0,
            method='Nelder-Mead',
            options={'maxiter': 50_000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True},
        )
        self._x0 = result.x   # warm-start next run
        return result.x

    # ── Reporting ────────────────────────────────────────────

    def report(self, p: np.ndarray):
        X  = params_to_mat(p)
        t, q = mat_to_xyzquat(X)

        # Best-estimate T_base_camera at each pose (for sanity check)
        residual = self._residual(p)

        msg = (
            f'\n{"="*55}\n'
            f' T_ee → camera (hand-eye transform)\n'
            f'{"="*55}\n'
            f'  translation  x={t[0]:+.6f}  y={t[1]:+.6f}  z={t[2]:+.6f}  [m]\n'
            f'  quaternion   x={q[0]:+.6f}  y={q[1]:+.6f}  z={q[2]:+.6f}  w={q[3]:+.6f}\n'
            f'  residual     {residual:.2e}   (lower is better; aim < 1e-3)\n'
            f'  recordings   {len(self.measurements)}\n'
            f'{"="*55}\n'
        )
        # Also print estimated marker pose in base frame
        poses = [Tb @ params_to_mat(p) @ Tc for Tb, Tc in self.measurements]
        marker_pos = np.mean([m[:3, 3] for m in poses], axis=0)
        msg += (
            f' Estimated marker position in base frame:\n'
            f'  x={marker_pos[0]:+.4f}  y={marker_pos[1]:+.4f}  z={marker_pos[2]:+.4f}  [m]\n'
            f'{"="*55}'
        )
        self.get_logger().info(msg)

    def save(self, p: np.ndarray):
        X = params_to_mat(p)
        t, q = mat_to_xyzquat(X)
        data = {
            'hand_eye_transform': {
                'parent_frame': self.ee_frame,
                'child_frame':  self.cam_frame,
                'translation': {'x': float(t[0]), 'y': float(t[1]), 'z': float(t[2])},
                'rotation':    {'x': float(q[0]), 'y': float(q[1]),
                                'z': float(q[2]), 'w': float(q[3])},
            }
        }
        with open(self.out_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        self.get_logger().info(f'Result saved → {self.out_file}')


# ──────────────────────────────────────────────────────────────
# Key input (raw, non-blocking wait)
# ──────────────────────────────────────────────────────────────

def getch() -> str:
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Hand-eye calibration via ArUco marker')
    p.add_argument('--base-frame',    default='fr3_link0',
                   help='Robot base TF frame  (default: fr3_link0)')
    p.add_argument('--ee-frame',      default='fr3_hand',
                   help='End-effector TF frame (default: fr3_hand)')
    p.add_argument('--camera-frame',  default='camera_color_optical_frame',
                   help='Camera optical frame  (default: camera_color_optical_frame)')
    p.add_argument('--marker-frame',  default='marker_0',
                   help='ArUco marker TF frame (default: marker_0)')
    p.add_argument('--out',           default='hand_eye_result.yaml',
                   help='Output YAML file      (default: hand_eye_result.yaml)')
    return p.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = HandEyeCalibrator(args)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    # Brief settle time for TF buffer to fill
    import time; time.sleep(1.0)

    print()
    print('╔══════════════════════════════════════════════════════╗')
    print('║         Hand-Eye Calibrator  (eye-in-hand)          ║')
    print('╠══════════════════════════════════════════════════════╣')
    print('║  R  — record current pose (make sure marker visible) ║')
    print('║  Q  — quit & save result                            ║')
    print('╚══════════════════════════════════════════════════════╝')
    print(f'  base  frame : {args.base_frame}')
    print(f'  ee    frame : {args.ee_frame}')
    print(f'  camera frame: {args.camera_frame}')
    print(f'  marker frame: {args.marker_frame}')
    print()
    print('Move robot to a NEW position/orientation, ensure the ArUco')
    print('marker is visible in the camera image, then press R.\n')

    last_p = None

    while rclpy.ok():
        print('Press R to record, Q to quit: ', end='', flush=True)
        key = getch()
        print(key.upper())

        if key.lower() == 'r':
            ok = node.record()
            if ok:
                p = node.optimise()
                if p is not None:
                    last_p = p
                    node.report(p)
                else:
                    print(f'  → {len(node.measurements)} measurement(s) so far (need ≥2 to estimate).')
            print()

        elif key.lower() == 'q':
            print('\nQuitting …')
            break

        else:
            print('  (unknown key — use R or Q)')

    if last_p is not None:
        node.report(last_p)
        node.save(last_p)
        t, q = mat_to_xyzquat(params_to_mat(last_p))
        print()
        print('── static_transform_publisher snippet ──────────────────')
        print(f'  ros2 run tf2_ros static_transform_publisher \\')
        print(f'    {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} \\')
        print(f'    {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f} \\')
        print(f'    {args.ee_frame} {args.camera_frame}')
        print('────────────────────────────────────────────────────────')
    else:
        print('Not enough data — no result saved.')

    rclpy.shutdown()


if __name__ == '__main__':
    main()
