"""Inference application — the backend-agnostic control loop (no training).

Ties the deployment layers into one loop that is identical for sim and hardware:

    read_state -> build obs -> [safety obs check] -> policy -> [safety approve] ->
    send joint targets -> log timing

The loop imports no simulator, no ROS, and no RL library — only the ``ActuatorInterface``
it is handed. Swapping ``SimBackend`` for ``Ros2Backend`` changes nothing here, which is
the whole point: the policy never learns whether it drives sim or metal.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from . import contract as C
from .action import ActionProcessor
from .hardware.base import ActuatorInterface, CommunicationError
from .observation import ObservationBuilder, RobotState
from .policy import PolicyRunner
from .safety import SafetyLayer, SafetyState


class LoopStats:
    """Rolling wall-clock timing for the control loop (ms)."""

    def __init__(self, window: int = 2000) -> None:
        self.cycle = deque(maxlen=window)

    def add_cycle(self, ms: float) -> None:
        self.cycle.append(ms)

    def snapshot(self) -> dict[str, float]:
        if not self.cycle:
            return {"count": 0}
        a = np.array(self.cycle)
        return {"count": len(a), "mean_ms": float(a.mean()),
                "p99_ms": float(np.percentile(a, 99)), "max_ms": float(a.max())}


class InferenceApp:
    """Deterministic policy inference driving an ActuatorInterface backend."""

    def __init__(self, policy_path: str, backend: ActuatorInterface,
                 safety: SafetyLayer | None = None, control_dt: float = 0.01) -> None:
        self.policy = PolicyRunner(policy_path)
        self.backend = backend
        self.obs_builder = ObservationBuilder()
        self.action_proc = ActionProcessor()      # standalone contract mapping (parity)
        self.safety = safety or SafetyLayer()
        self.control_dt = control_dt
        self.loop_stats = LoopStats()
        #: (3,) base-frame reach target (metres). Default = the fixed diagnostic point.
        self._target = np.array([0.40, 0.20, 0.30], dtype=np.float32)
        self._last_clipped = np.zeros(C.ACTION_DIM, dtype=np.float32)

    def set_target(self, target_xyz: np.ndarray) -> None:
        """Set the reach target (base frame, metres)."""
        self._target = np.asarray(target_xyz, dtype=np.float32).reshape(3)

    def step(self) -> dict:
        """Run one control cycle. Returns a per-cycle telemetry dict."""
        t0 = time.perf_counter()
        try:
            state: RobotState = self.backend.read_state()
        except CommunicationError:
            self.safety.trip("comm_error")
            return {"fault": "comm_error", "safety": self.safety.state.value}

        obs = self.obs_builder.build(state, self._target, self._last_clipped)
        self.safety.check_observation(obs)
        self.safety.note_inference()

        raw = self.policy.infer(obs)
        report = self.safety.approve(raw, state.joint_pos, self.control_dt, state.stamp)
        self._last_clipped = report.clipped_action   # single source for next last_action

        self.backend.send_joint_targets(report.target)

        cycle_ms = (time.perf_counter() - t0) * 1e3
        self.loop_stats.add_cycle(cycle_ms)
        dist = float(np.linalg.norm(self._target - state.ee_position))
        return {"safety": report.state.value, "holding": report.holding,
                "faults": report.faults, "distance_m": dist,
                "infer_ms": self.policy.latency.snapshot().get("mean_ms", float("nan")),
                "cycle_ms": cycle_ms}

    def run(self, steps: int, verbose_every: int = 100) -> dict:
        """Run ``steps`` control cycles at the target rate; return a timing summary."""
        self.safety.reset()
        for i in range(steps):
            info = self.step()
            if verbose_every and (i % verbose_every == 0 or i == steps - 1):
                print(f"[infer] step {i:>5}  safety={info.get('safety')}  "
                      f"dist={info.get('distance_m', float('nan'))*100:.1f}cm  "
                      f"infer={info.get('infer_ms', float('nan')):.2f}ms  "
                      f"cycle={info.get('cycle_ms', float('nan')):.2f}ms", flush=True)
            if self.safety.state is SafetyState.ESTOP:
                print("[infer] E-STOP latched; stopping loop.", flush=True)
                break
        return {"inference_latency": self.policy.latency.snapshot(),
                "loop_timing": self.loop_stats.snapshot(),
                "final_safety": self.safety.state.value}
