"""Simulation backend — drives the Isaac Sim reach articulation as an ActuatorInterface.

This lets the *exact same* inference application run against the simulator: it commands
final joint position targets to the robot articulation and reads back joint/EE state,
bypassing the training env's ActionManager (the deployment ActionProcessor already did
the scale/offset/clip). That mirrors what a real driver does — receive final targets,
report state — so a passing sim run exercises the whole deployment path end-to-end.

Isaac Lab is imported lazily inside ``connect()`` so the deployment package stays
importable on a hardware machine with no simulator.
"""

from __future__ import annotations

import time

import numpy as np

from .. import contract as C
from ..observation import RobotState
from .base import ActuatorInterface, CommunicationError


class SimBackend(ActuatorInterface):
    """ActuatorInterface backed by a single-env Isaac Lab reach scene.

    Requires the Isaac Sim app to already be launched (the inference app / verification
    script owns the ``AppLauncher``). Reads joint pos/vel and EE from the articulation;
    ``send_joint_targets`` sets position targets and advances the physics one control step.
    """

    name = "sim"

    def __init__(self, task_id: str | None = None, device: str = "cpu") -> None:
        self._task_id = task_id
        self._device = device
        self._env = None
        self._robot = None
        self._ee_body_id = None
        self._dt = 1.0 / 100.0   # control dt (decimation 2 @ 200 Hz), overwritten on connect
        self._t = 0.0

    def connect(self) -> None:
        import tasks  # noqa: F401  (registers the task)
        from runners.env_loader import make_reach_env

        task_id = self._task_id or tasks.REACH_TASK_ID
        self._env = make_reach_env(task_id, num_envs=1, device=self._device, seed=42)
        self._env.reset()               # reset via the gym wrapper (order-enforcing)
        env = self._env.unwrapped
        self._robot = env.scene["robot"]
        self._ee_body_id = self._robot.find_bodies(C.EE_BODY_NAME)[0][0]
        self._decimation = int(env.cfg.decimation)             # physics steps / control step
        self._physics_dt = float(env.cfg.sim.dt)
        self._dt = self._physics_dt * self._decimation          # control-step dt
        self._t = 0.0

    def disconnect(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
                self._robot = None

    @property
    def control_dt(self) -> float:
        return self._dt

    def _env_origin(self) -> np.ndarray:
        return self._env.unwrapped.scene.env_origins[0].cpu().numpy()

    def read_state(self) -> RobotState:
        if self._robot is None:
            raise CommunicationError("SimBackend not connected")
        data = self._robot.data
        jp = data.joint_pos.torch[0].cpu().numpy().astype(np.float32)
        jv = data.joint_vel.torch[0].cpu().numpy().astype(np.float32)
        ee_w = data.body_pos_w.torch[0, self._ee_body_id].cpu().numpy().astype(np.float32)
        ee = ee_w - self._env_origin()   # base/env frame, matching the training obs
        return RobotState(joint_pos=jp[:C.NUM_JOINTS], joint_vel=jv[:C.NUM_JOINTS],
                          ee_position=ee.astype(np.float32), stamp=self._t)

    def send_joint_targets(self, targets: np.ndarray) -> None:
        """Actuate the arm to ``targets`` (rad) via the validated env pipeline.

        We drive through ``env.step`` (which runs the action manager + actuators +
        decimation exactly as in training) instead of hand-rolling the physics loop.
        The ManagerBasedEnv action is the normalized action, so we invert the contract
        mapping: ``raw = (target - home) / scale`` (already clipped/limited upstream). Net
        effect: the arm tracks the commanded joint targets through the proven controller.
        """
        if self._robot is None:
            raise CommunicationError("SimBackend not connected")
        import torch
        raw = np.clip((np.asarray(targets, dtype=np.float32) - C.HOME_POSE) / C.ACTION_SCALE,
                      -C.CLIP_ACTIONS, C.CLIP_ACTIONS)
        action = torch.as_tensor(raw, device=self._device).unsqueeze(0)
        self._env.step(action)   # gym ManagerBasedRLEnv.step: action-mgr + actuators + decimation
        self._t += self._dt
