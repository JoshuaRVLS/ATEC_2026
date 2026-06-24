"""Task D B2Piper controller: velocity-command replay through the B2 policy.

This file intentionally stays small.  Task D box contact is too inconsistent for
the previous open-loop state machine, so the deadline path is:
  1. drive manually with scripts/play_atec_task.py --teleop-record,
  2. save the best command macro as demo/task_d_manual_best.json,
  3. replay those velocity commands through the existing locomotion policy.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch


class AlgSolution:
    """Replay/manual velocity commands while the B2 locomotion policy stabilizes legs."""

    DEFAULT_REPLAY_NAME = "task_d_manual_best.json"
    REPLAY_END_STOP_STEPS = 50
    REPLAY_END_SLOW_FORWARD = 0.25

    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        policy_path = base_dir / "policy.pt"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy = torch.jit.load(str(policy_path), map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim), device=self.device, dtype=torch.float32
        )

        self._dt = 0.02
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0

        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_z = 0.0
        self.step = 0
        self.phase = "REPLAY"
        self._printed_obs = False

        self.manual_control = False
        self.manual_world_frame = False
        self.manual_vx = 0.0
        self.manual_vy = 0.0
        self.manual_wz = 0.0

        self.replay_commands: list[dict[str, Any]] = []
        self.replay_index = 0
        self.replay_end_index: int | None = None
        self._replay_warned = False
        self._load_default_replay(base_dir / self.DEFAULT_REPLAY_NAME)

    # ------------------------------------------------------------------
    # Public hooks used by scripts/play_atec_task.py
    # ------------------------------------------------------------------

    def set_manual_command(
        self, vx: float, vy: float, wz: float, world_frame: bool = False
    ) -> None:
        self.manual_control = True
        self.manual_world_frame = bool(world_frame)
        self.manual_vx = float(vx)
        self.manual_vy = float(vy)
        self.manual_wz = float(wz)

    def clear_manual_command(self) -> None:
        self.manual_control = False
        self.manual_world_frame = False
        self.manual_vx = 0.0
        self.manual_vy = 0.0
        self.manual_wz = 0.0

    def load_replay(self, path: str | os.PathLike[str]) -> None:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        commands = payload.get("commands", payload) if isinstance(payload, dict) else payload
        self.set_replay_commands(commands)

    def set_replay_commands(self, commands: list[Any]) -> None:
        self.replay_commands = [self._normalize_command(cmd) for cmd in commands]
        self.replay_index = 0
        self.replay_end_index = None
        self._replay_warned = False

    # ------------------------------------------------------------------
    # Command source
    # ------------------------------------------------------------------

    def _load_default_replay(self, path: Path) -> None:
        if path.exists():
            self.load_replay(path)
            print(f"[REPLAY] loaded {len(self.replay_commands)} commands from {path}")
        else:
            print(f"[REPLAY] no default replay at {path}; standing by.")

    @staticmethod
    def _normalize_command(cmd: Any) -> dict[str, Any]:
        if isinstance(cmd, dict):
            return {
                "vx": float(cmd.get("vx", 0.0)),
                "vy": float(cmd.get("vy", 0.0)),
                "wz": float(cmd.get("wz", 0.0)),
                "world_frame": bool(cmd.get("world_frame", False)),
                "dt": float(cmd.get("dt", 0.02)),
            }
        if isinstance(cmd, (list, tuple)) and len(cmd) >= 3:
            return {
                "vx": float(cmd[0]),
                "vy": float(cmd[1]),
                "wz": float(cmd[2]),
                "world_frame": bool(cmd[3]) if len(cmd) >= 4 else False,
                "dt": float(cmd[4]) if len(cmd) >= 5 else 0.02,
            }
        raise ValueError(f"Invalid replay command: {cmd!r}")

    def _next_replay_command(self) -> tuple[float, float, float, bool, str]:
        if self.replay_index < len(self.replay_commands):
            cmd = self.replay_commands[self.replay_index]
            self.replay_index += 1
            return cmd["vx"], cmd["vy"], cmd["wz"], cmd["world_frame"], "REPLAY"

        if not self.replay_commands:
            if not self._replay_warned:
                print("[REPLAY] no commands loaded; outputting zero velocity.")
                self._replay_warned = True
            return 0.0, 0.0, 0.0, False, "NO_REPLAY"

        if self.replay_end_index is None:
            self.replay_end_index = self.step
            print("[REPLAY] finished; stopping briefly, then slow-forward fallback.")

        after_end = self.step - self.replay_end_index
        if after_end < self.REPLAY_END_STOP_STEPS:
            return 0.0, 0.0, 0.0, False, "REPLAY_END_STOP"
        return self.REPLAY_END_SLOW_FORWARD, 0.0, 0.0, False, "REPLAY_END_FORWARD"

    # ------------------------------------------------------------------
    # Pose/velocity command helpers
    # ------------------------------------------------------------------

    def _update_pose(self, proprio: torch.Tensor) -> None:
        base_lin = proprio[0, 0:3].detach().cpu().numpy()
        base_ang = proprio[0, 3:6].detach().cpu().numpy()
        vx_body, vy_body = float(base_lin[0]), float(base_lin[1])
        yaw_rate = float(base_ang[2])

        cos_y = math.cos(self.est_yaw)
        sin_y = math.sin(self.est_yaw)
        self.est_x += (cos_y * vx_body - sin_y * vy_body) * self._dt
        self.est_y += (sin_y * vx_body + cos_y * vy_body) * self._dt
        self.est_yaw += yaw_rate * self._dt

        while self.est_yaw > math.pi:
            self.est_yaw -= 2.0 * math.pi
        while self.est_yaw < -math.pi:
            self.est_yaw += 2.0 * math.pi

    def _set_body_velocity(self, vx: float, vy: float, wz: float) -> None:
        self._vel_x = float(vx)
        self._vel_y = float(vy)
        self._vel_z = float(wz)

    def _set_world_velocity(self, vx: float, vy: float, wz: float) -> None:
        cos_y = math.cos(self.est_yaw)
        sin_y = math.sin(self.est_yaw)
        self._vel_x = cos_y * vx + sin_y * vy
        self._vel_y = -sin_y * vx + cos_y * vy
        self._vel_z = float(wz)

    def _apply_command(self, vx: float, vy: float, wz: float, world_frame: bool) -> None:
        if world_frame:
            self._set_world_velocity(vx, vy, wz)
        else:
            self._set_body_velocity(vx, vy, wz)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        num_envs = int(proprio.shape[0])
        cmd = torch.tensor(
            [self._vel_x, self._vel_y, self._vel_z],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd.to(dtype=proprio.dtype)

    def _extract_policy_obs(self, obs, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        idx = 0
        _ = proprio[:, idx:idx + 3]; idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        _ = proprio[:, idx:idx + 3]; idx += 3
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]
        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)

        return torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                self._get_velocity_commands(proprio),
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(f"Expected {self.leg_action_dim}, got {action_train.shape[-1]}")

        num_envs = action_train.shape[0]
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = action_train * self.train_to_env_action_scale
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def _run_policy(self, obs, action_dim: int) -> torch.Tensor:
        policy_obs = self._extract_policy_obs(obs, action_dim)
        with torch.inference_mode():
            action_train = self.policy(policy_obs)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        return self._map_policy_action_to_env_action(action_train, action_dim)

    def predicts(self, obs, current_score):
        if not self._printed_obs:
            print("OBS KEYS:", list(obs.keys()))
            self._printed_obs = True

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        self._update_pose(proprio)

        if self.manual_control:
            vx, vy, wz = self.manual_vx, self.manual_vy, self.manual_wz
            world_frame = self.manual_world_frame
            self.phase = "TELEOP"
        else:
            vx, vy, wz, world_frame, self.phase = self._next_replay_command()

        self._apply_command(vx, vy, wz, world_frame)
        action = self._run_policy(obs, action_dim)

        if self.step % 25 == 0:
            frame = "W" if world_frame else "B"
            print(
                f"[D]{self.phase:<18}|{self.step:<4}|cmd{frame}=({vx:+.2f},{vy:+.2f},{wz:+.2f})|"
                f"robot=({self.est_x:+.1f},{self.est_y:+.1f},{math.degrees(self.est_yaw):+.0f}deg)"
            )

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}
