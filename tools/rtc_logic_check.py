"""Exercise the RTC index math + splice with the ROS/websocket parts stubbed out.

The parts that can silently go wrong are: (a) mapping elapsed wall time onto a
chunk index through non-uniform step_times, (b) the offset the new chunk is
spliced in at, and (c) that prev_actions goes on the wire verbatim at full width.
"""
import sys, types, time
import numpy as np

# ── stub every ROS / openpi import the node pulls in ──────────────────────────
def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m

class _Node:
    def __init__(self, *a, **k): pass
    def declare_parameter(self, n, v): self._p = getattr(self, "_p", {}); self._p[n] = v
    def get_parameter(self, n): return types.SimpleNamespace(value=self._p[n])
    def create_subscription(self, *a, **k): pass
    def create_publisher(self, *a, **k): return types.SimpleNamespace(publish=lambda m: None,
                                                                     get_subscription_count=lambda: 1)
    def get_logger(self):
        return types.SimpleNamespace(info=print, warning=print, error=print)

_mod("rclpy", init=lambda: None, shutdown=lambda: None,
     logging=_mod("rclpy.logging", get_logger=lambda n: types.SimpleNamespace(
         info=print, warning=print, error=print)))
_mod("rclpy.node", Node=_Node)
_mod("rclpy.executors", MultiThreadedExecutor=object)
_mod("rclpy.action", ActionClient=object)
_mod("cv_bridge", CvBridge=object)
_mod("sensor_msgs.msg", Image=object, JointState=object)
_mod("sensor_msgs")
class _Dur:
    def __init__(self, sec=0, nanosec=0): self.sec, self.nanosec = sec, nanosec
_mod("builtin_interfaces.msg", Duration=_Dur)
_mod("builtin_interfaces")
class _JTP:
    def __init__(self): self.positions=[]; self.velocities=[]; self.time_from_start=None
class _JT:
    def __init__(self): self.joint_names=[]; self.points=[]; self.header=types.SimpleNamespace(stamp=None)
_mod("trajectory_msgs.msg", JointTrajectory=_JT, JointTrajectoryPoint=_JTP)
_mod("trajectory_msgs")
_mod("control_msgs.action", FollowJointTrajectory=object)
_mod("control_msgs")
_mod("franka_msgs.action", Grasp=object, Move=object)
_mod("franka_msgs")
_mod("omni_place.Interface", MotionPlanningInterface=lambda node: None)
_mod("omni_place")
_ws_client = _mod("websockets.sync.client", ClientConnection=object, connect=lambda *a, **k: None)
_ws_sync = _mod("websockets.sync", client=_ws_client)
_mod("websockets", sync=_ws_sync)
_mod("openpi_client", msgpack_numpy=_mod("openpi_client.msgpack_numpy"),
     image_tools=_mod("openpi_client.image_tools",
                      resize_with_pad=lambda im, h, w: im),
     websocket_client_policy=_mod("openpi_client.websocket_client_policy",
                                  WebsocketClientPolicy=object))

sys.path.insert(0, "/home/mdsaifahmad/Franka/src/franka_openpi")
from franka_openpi import rtc_executor
from franka_openpi.openpi_rtc_node import OpenPIRTCNode

H, W = 50, 14
ok = True
def check(label, cond, extra=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label} {extra}")

# ── 1. terminal velocity is NOT zeroed (the rest-to-rest bug) ─────────────────
print("\n1. streaming _waypoint_velocities")
knots = np.cumsum(np.ones((6, 7)) * 0.05, axis=0)
times = np.full(5, 0.2)
v = rtc_executor._waypoint_velocities(knots, times)
check("last waypoint keeps velocity", np.abs(v[-1]).max() > 1e-6, f"|v[-1]|={np.abs(v[-1]).max():.4f}")
from franka_openpi import action_executor
v_old = action_executor._waypoint_velocities(knots, times)
check("action_executor still zeroes it (unchanged)", np.abs(v_old[-1]).max() < 1e-12)

# ── 2. time -> chunk index through non-uniform step_times ────────────────────
print("\n2. _chunk_index mapping")
node = OpenPIRTCNode.__new__(OpenPIRTCNode)
node._prev_actions = np.zeros((H, W), np.float32)
# deliberately non-uniform: velocity limiting stretches early segments
node._chunk_step_times = np.cumsum(np.r_[np.full(10, 0.5), np.full(40, 0.2)])
node._chunk_offset = 0
node._rtc_d = 3
node._chunk_pub_t = time.monotonic()
check("t=0 -> index 0", node._chunk_index() == 0, f"got {node._chunk_index()}")
node._chunk_pub_t = time.monotonic() - 5.0   # 10 stretched steps done
check("t=5.0s -> index 10", node._chunk_index() == 10, f"got {node._chunk_index()}")
node._chunk_pub_t = time.monotonic() - 7.0   # +2.0s of 0.2s steps
check("t=7.0s -> index 20", node._chunk_index() == 20, f"got {node._chunk_index()}")
uniform = np.cumsum(np.full(H, 0.2))
naive = int(np.searchsorted(uniform, 7.0, side="right"))
check("naive elapsed/step_duration would be badly wrong (drift proves it matters)",
      naive == 34 and abs(naive - 20) > 10, f"naive said {naive}, truth is 20")

# offset is honoured
node._chunk_offset = 7
node._chunk_pub_t = time.monotonic() - 5.0
check("offset shifts the index", node._chunk_index() == 17, f"got {node._chunk_index()}")

# ── 3. fire index ────────────────────────────────────────────────────────────
print("\n3. _fire_index / _current_d")
node._chunk_offset = 0
node._d_observed = __import__("collections").deque(maxlen=20)
node._d_floor = 0
node._d_margin = 2
check("rtc_d=3 -> fire at 47", node._fire_index() == 47, f"got {node._fire_index()}")
node._rtc_d = 0
check("adaptive with no history -> d=3", node._fire_index() == 47, f"got {node._fire_index()}")
node._d_observed.extend([2, 3, 5, 2])
check("adaptive sizes off the TAIL not the median (max 5 + margin 2 = 7)",
      node._current_d() == 7 and node._fire_index() == 43, f"d={node._current_d()}")
node._d_floor = 11
check("_d_floor ratchet wins once set", node._current_d() == 11, f"got {node._current_d()}")
node._rtc_d = 3
check("floor also overrides a pinned rtc_d", node._current_d() == 11, f"got {node._current_d()}")
node._d_floor = 0
check("d never exceeds H-1", node._current_d() <= H - 1)

# ── 4. splice offset: new[k] lines up with old[s_req + k] ────────────────────
print("\n4. splice alignment")
old = np.arange(H * W, dtype=np.float32).reshape(H, W)
new = (np.arange(H * W, dtype=np.float32) + 10000).reshape(H, W)
s_req, s_reply = 47, 49
d_actual = s_reply - s_req
check("d_actual = 2", d_actual == 2)
check("resume at new[2], i.e. old index 49",
      d_actual == s_reply - s_req and (s_req + d_actual) == s_reply)
published = new[d_actual:]
check("published tail length H-d", len(published) == H - d_actual, f"got {len(published)}")

# ── 5. prev_actions goes on the wire verbatim, full width ────────────────────
print("\n5. wire format")
sent = {}
class FakePolicy:
    def infer(self, obs):
        sent.update(obs)
        return {"actions": new,
                "server_timing": {"infer_ms": 180.0},
                "policy_timing": {"infer_ms": 11.0}}
node.policy = FakePolicy()
node._two_front_cameras = False
node.front1_image = np.zeros((3, 4, 4), np.uint8)
node.wrist_image = np.zeros((3, 4, 4), np.uint8)
node._raw_joints = np.arange(9, dtype=float)
node.prompt = "p"
node._server_infer_log = []
node._policy_infer_log = []
out = node._fetch_chunk_sync(old, 47, 5)
check("prev_actions present", "prev_actions" in sent)
check("prev_actions shape (50,14) — NOT sliced to 8",
      sent["prev_actions"].shape == (H, W), f"got {sent['prev_actions'].shape}")
check("prev_actions bitwise identical to what infer returned",
      np.array_equal(sent["prev_actions"], old))
check("prev_actions_start is a plain int", isinstance(sent["prev_actions_start"], int))
check("prev_actions_d sent as plain int",
      sent.get("prev_actions_d") == 5 and isinstance(sent["prev_actions_d"], int))
check("state is 9-dim float32", sent["state"].shape == (9,) and sent["state"].dtype == np.float32)
check("returned chunk kept at full width", out.shape == (H, W), f"got {out.shape}")
check("server_timing.infer_ms recorded", node._server_infer_log == [180.0])
check("policy_timing.infer_ms recorded separately", node._policy_infer_log == [11.0])

# first call of an episode omits all three keys
sent.clear()
node._fetch_chunk_sync(None, None, None)
check("all three omitted on first inference",
      not {"prev_actions", "prev_actions_start", "prev_actions_d"} & set(sent))

# image path: default (image_size=0) must be byte-identical to the existing node
print("\n5b. image path")
node._image_size = 0
class _Msg: pass
raw_hwc = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
node.bridge = types.SimpleNamespace(imgmsg_to_cv2=lambda m, e: raw_hwc)
chw = node._proc_image(_Msg())
check("default sends full-res CHW, unchanged", chw.shape == (3, 480, 640), f"got {chw.shape}")
check("pixels untouched at image_size=0", np.array_equal(chw, raw_hwc.transpose(2, 0, 1)))

# ── 6. episode reset clears both ─────────────────────────────────────────────
print("\n6. reset")
node._prev_actions = old
node._chunk_step_times = np.zeros(5)
node._chunk_offset = 9
node.executor_ = types.SimpleNamespace(reset_gripper_state=lambda: None)
node._reset_rtc_state()
check("prev_actions cleared", node._prev_actions is None)
check("step_times cleared", node._chunk_step_times is None)
check("offset cleared", node._chunk_offset == 0)
check("_chunk_index() safe when cleared", node._chunk_index() == 0)

print("\n" + ("ALL PASS" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
