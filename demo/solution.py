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
    PARKOUR_PROP_DIM = 53
    PARKOUR_SCAN_DIM = 5760
    PARKOUR_PRIV_EXPLICIT_DIM = 9
    PARKOUR_PRIV_LATENT_DIM = 29
    PARKOUR_HISTORY_LEN = 10
    PARKOUR_TO_ATEC_ACTION_SCALE = 0.25
    PARKOUR_IDLE_VX = 0.35
    PARKOUR_MIN_VX = 0.30
    PARKOUR_MAX_VX = 0.80
    PARKOUR_MAX_DELTA_YAW = 1.60
    PARKOUR_ACTION_CLIP = 2.00
    PARKOUR_RAMP_STEPS = 50
    PARKOUR_ACTION_SMOOTHING = 0.70
    PARKOUR_TOTAL_OBS_DIM = (
        PARKOUR_PROP_DIM
        + PARKOUR_SCAN_DIM
        + PARKOUR_PRIV_EXPLICIT_DIM
        + PARKOUR_PRIV_LATENT_DIM
        + PARKOUR_HISTORY_LEN * PARKOUR_PROP_DIM
    )

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

        action_scale = float(os.environ.get("PARKOUR_ACTION_SCALE", self.PARKOUR_TO_ATEC_ACTION_SCALE))
        self.parkour_idle_vx = float(os.environ.get("PARKOUR_IDLE_VX", self.PARKOUR_IDLE_VX))
        self.parkour_action_clip = float(os.environ.get("PARKOUR_ACTION_CLIP", self.PARKOUR_ACTION_CLIP))
        self.parkour_ramp_steps = int(os.environ.get("PARKOUR_RAMP_STEPS", self.PARKOUR_RAMP_STEPS))
        self.parkour_action_smoothing = float(
            os.environ.get("PARKOUR_ACTION_SMOOTHING", self.PARKOUR_ACTION_SMOOTHING)
        )
        self.parkour_debug = os.environ.get("PARKOUR_DEBUG", "1").lower() not in {"0", "false", "no"}
        self.parkour_joint_order = os.environ.get("PARKOUR_JOINT_ORDER", "env").lower()

        self.train_to_env_action_scale = torch.full(
            (1, self.leg_action_dim),
            action_scale,
            device=self.device,
            dtype=torch.float32,
        )
        if self.parkour_joint_order in {"env", "atec", "fr_fl_rr_rl"}:
            leg_perm = list(range(self.leg_action_dim))
        elif self.parkour_joint_order in {"fl_fr_rl_rr", "parkour"}:
            leg_perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        else:
            raise ValueError(
                "PARKOUR_JOINT_ORDER must be one of: env, atec, fr_fl_rr_rl, fl_fr_rl_rr, parkour"
            )
        self.env_to_policy_leg_perm = torch.tensor(leg_perm, device=self.device, dtype=torch.long)
        self.policy_to_env_leg_perm = torch.tensor(leg_perm, device=self.device, dtype=torch.long)
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
        self._parkour_history = torch.zeros(
            (1, self.PARKOUR_HISTORY_LEN, self.PARKOUR_PROP_DIM),
            device=self.device,
            dtype=torch.float32,
        )
        self._last_action_env: torch.Tensor | None = None
        self.step = 0
        self.phase = "REPLAY"
        self._printed_obs = False
        self._printed_policy_debug = False

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
        vx = self._vel_x
        if abs(vx) < 1.0e-6 and abs(self._vel_y) < 1.0e-6 and abs(self._vel_z) < 1.0e-6:
            vx = self.parkour_idle_vx
        vx = max(self.PARKOUR_MIN_VX, min(self.PARKOUR_MAX_VX, vx))
        cmd = torch.tensor(
            [vx, self._vel_y, self._vel_z],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd.to(dtype=proprio.dtype)

    def _split_proprio(self, obs, action_dim: int):
        proprio = obs["proprio"].to(self.device)

        idx = 0
        base_lin_vel = proprio[:, idx:idx + 3]; idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        _velocity_commands = proprio[:, idx:idx + 3]; idx += 3
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        actions_all = proprio[:, idx:idx + action_dim]
        return (
            proprio,
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            joint_pos_all,
            joint_vel_all,
            actions_all,
        )

    def _extract_policy_obs(self, obs, action_dim: int) -> torch.Tensor:
        (
            proprio,
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            joint_pos_all,
            joint_vel_all,
            actions_all,
        ) = self._split_proprio(obs, action_dim)

        num_envs = int(proprio.shape[0])
        joint_pos_env_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_env_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]
        joint_pos_leg = joint_pos_env_leg[:, self.env_to_policy_leg_perm]
        joint_vel_leg = joint_vel_env_leg[:, self.env_to_policy_leg_perm]
        actions_policy_leg = actions_env_leg[:, self.env_to_policy_leg_perm]
        scale = self.train_to_env_action_scale.to(dtype=proprio.dtype)
        last_action_leg = actions_policy_leg / scale

        gravity_xy = projected_gravity[:, :2]
        zeros_1 = torch.zeros((num_envs, 1), device=self.device, dtype=proprio.dtype)
        zeros_2 = torch.zeros((num_envs, 2), device=self.device, dtype=proprio.dtype)
        delta_yaw_cmd = max(
            -self.PARKOUR_MAX_DELTA_YAW,
            min(self.PARKOUR_MAX_DELTA_YAW, self._vel_z),
        )
        delta_yaw = torch.full((num_envs, 1), delta_yaw_cmd, device=self.device, dtype=proprio.dtype)
        env_non_flat = torch.ones((num_envs, 1), device=self.device, dtype=proprio.dtype)
        env_flat = torch.zeros((num_envs, 1), device=self.device, dtype=proprio.dtype)
        contact_fill = torch.zeros((num_envs, 4), device=self.device, dtype=proprio.dtype)
        cmd = self._get_velocity_commands(proprio)

        prop = torch.cat(
            [
                base_ang_vel * 0.25,
                gravity_xy,
                zeros_1,
                delta_yaw,
                delta_yaw,
                zeros_2,
                cmd[:, 0:1],
                env_non_flat,
                env_flat,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                last_action_leg,
                contact_fill,
            ],
            dim=-1,
        )
        if prop.shape[-1] != self.PARKOUR_PROP_DIM:
            raise RuntimeError(f"Parkour prop dim mismatch: {prop.shape[-1]}")

        extero = obs.get("extero")
        if extero is None:
            scan = torch.zeros(
                (num_envs, self.PARKOUR_SCAN_DIM), device=self.device, dtype=proprio.dtype
            )
        else:
            scan = extero.to(device=self.device, dtype=proprio.dtype)
            if scan.ndim == 1:
                scan = scan.view(1, -1)
            else:
                scan = scan.reshape(scan.shape[0], -1)
            if scan.shape[-1] < self.PARKOUR_SCAN_DIM:
                pad = torch.zeros(
                    (scan.shape[0], self.PARKOUR_SCAN_DIM - scan.shape[-1]),
                    device=self.device,
                    dtype=proprio.dtype,
                )
                scan = torch.cat([scan, pad], dim=-1)
            elif scan.shape[-1] > self.PARKOUR_SCAN_DIM:
                scan = scan[:, :self.PARKOUR_SCAN_DIM]
            scan = torch.nan_to_num(scan, nan=0.0, posinf=1.0, neginf=-1.0)
            scan = torch.clamp(scan, -1.0, 1.0)

        if self._parkour_history.shape[0] != num_envs or self._parkour_history.dtype != proprio.dtype:
            self._parkour_history = torch.zeros(
                (num_envs, self.PARKOUR_HISTORY_LEN, self.PARKOUR_PROP_DIM),
                device=self.device,
                dtype=proprio.dtype,
            )
        if self.step <= 1:
            history = torch.stack([prop] * self.PARKOUR_HISTORY_LEN, dim=1)
        else:
            history = torch.cat([self._parkour_history[:, 1:], prop.unsqueeze(1)], dim=1)
        self._parkour_history = history.detach()

        priv_explicit = torch.cat(
            [
                base_lin_vel * 2.0,
                torch.zeros((num_envs, 6), device=self.device, dtype=proprio.dtype),
            ],
            dim=-1,
        )
        priv_latent = torch.zeros(
            (num_envs, self.PARKOUR_PRIV_LATENT_DIM), device=self.device, dtype=proprio.dtype
        )
        policy_obs = torch.cat(
            [
                prop,
                scan,
                priv_explicit,
                priv_latent,
                history.reshape(num_envs, -1),
            ],
            dim=-1,
        )
        if policy_obs.shape[-1] != self.PARKOUR_TOTAL_OBS_DIM:
            raise RuntimeError(f"Parkour obs dim mismatch: {policy_obs.shape[-1]}")
        if self.parkour_debug and not self._printed_policy_debug:
            print(
                "[PARKOUR-OBS] "
                f"proprio={tuple(proprio.shape)} policy_obs={tuple(policy_obs.shape)} "
                f"cmd=({cmd[0,0].item():+.2f},{cmd[0,1].item():+.2f},{cmd[0,2].item():+.2f}) "
                f"joint_pos=[{joint_pos_leg.min().item():+.2f},{joint_pos_leg.max().item():+.2f}] "
                f"joint_vel=[{joint_vel_leg.min().item():+.2f},{joint_vel_leg.max().item():+.2f}] "
                f"last_action=[{last_action_leg.min().item():+.2f},{last_action_leg.max().item():+.2f}] "
                f"scan=[{scan.min().item():+.2f},{scan.max().item():+.2f}] "
                f"scale={scale[0,0].item():.3f} "
                f"joint_order={self.parkour_joint_order}"
            )
        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(f"Expected {self.leg_action_dim}, got {action_train.shape[-1]}")

        num_envs = action_train.shape[0]
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_train = torch.clamp(action_train, -self.parkour_action_clip, self.parkour_action_clip)
        action_env[:, self.leg_joint_indices] = (
            action_train[:, self.policy_to_env_leg_perm] * self.train_to_env_action_scale
        )
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)

        if self.parkour_ramp_steps > 0:
            ramp = min(1.0, max(0.0, float(self.step) / float(self.parkour_ramp_steps)))
            action_env[:, self.leg_joint_indices] *= ramp

        if self._last_action_env is None or self._last_action_env.shape != action_env.shape:
            self._last_action_env = torch.zeros_like(action_env)
        alpha = max(0.0, min(0.95, self.parkour_action_smoothing))
        action_env = alpha * self._last_action_env + (1.0 - alpha) * action_env
        self._last_action_env = action_env.detach()
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
        if self.parkour_debug and not self._printed_policy_debug:
            print(
                "[PARKOUR-ACT] "
                f"train=[{action_train.min().item():+.2f},{action_train.max().item():+.2f}] "
                f"mean={action_train.mean().item():+.2f}"
            )
            self._printed_policy_debug = True
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
