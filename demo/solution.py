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
    # Read from demo/policy.pt TorchScript archive:
    # num_prop=48, num_scan=5760, num_priv_explicit=0, num_priv_latent=0, num_hist=0.
    # Keep these tied to the exported model, not the generic parkour source config.
    PARKOUR_PROP_DIM = 48
    PARKOUR_SCAN_DIM = 5760
    PARKOUR_PRIV_EXPLICIT_DIM = 0
    PARKOUR_PRIV_LATENT_DIM = 0
    PARKOUR_HISTORY_LEN = 0
    PARKOUR_TO_ATEC_ACTION_SCALE = 0.5
    PARKOUR_IDLE_VX = 0.35
    PARKOUR_MIN_VX = 0.30
    PARKOUR_MAX_VX = 0.80
    PARKOUR_ACTION_CLIP = 4.80
    PARKOUR_RAMP_STEPS = 0
    PARKOUR_START_FACTOR = 1.00
    PARKOUR_ACTION_SMOOTHING = 0.00
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

        self.parkour_prop_dim = self._policy_int_attr("num_prop", self.PARKOUR_PROP_DIM)
        self.parkour_scan_dim = self._policy_int_attr("num_scan", self.PARKOUR_SCAN_DIM)
        self.parkour_priv_explicit_dim = self._policy_int_attr(
            "num_priv_explicit", self.PARKOUR_PRIV_EXPLICIT_DIM
        )
        self.parkour_priv_latent_dim = self._policy_int_attr(
            "num_priv_latent", self.PARKOUR_PRIV_LATENT_DIM
        )
        self.parkour_history_len = self._policy_int_attr("num_hist", self.PARKOUR_HISTORY_LEN)
        self.parkour_total_obs_dim = (
            self.parkour_prop_dim
            + self.parkour_scan_dim
            + self.parkour_priv_explicit_dim
            + self.parkour_priv_latent_dim
            + self.parkour_history_len * self.parkour_prop_dim
        )

        self.leg_action_dim = self._policy_int_attr("num_actions", 12)
        self.arm_action_dim = 8
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        action_scale = float(os.environ.get("PARKOUR_ACTION_SCALE", self.PARKOUR_TO_ATEC_ACTION_SCALE))
        self.parkour_idle_vx = float(os.environ.get("PARKOUR_IDLE_VX", self.PARKOUR_IDLE_VX))
        self.parkour_min_vx = float(os.environ.get("PARKOUR_MIN_VX", self.PARKOUR_MIN_VX))
        self.parkour_max_vx = float(os.environ.get("PARKOUR_MAX_VX", self.PARKOUR_MAX_VX))
        self.parkour_action_clip = float(os.environ.get("PARKOUR_ACTION_CLIP", self.PARKOUR_ACTION_CLIP))
        self.parkour_ramp_steps = int(os.environ.get("PARKOUR_RAMP_STEPS", self.PARKOUR_RAMP_STEPS))
        self.parkour_start_factor = float(os.environ.get("PARKOUR_START_FACTOR", self.PARKOUR_START_FACTOR))
        self.parkour_action_smoothing = float(
            os.environ.get("PARKOUR_ACTION_SMOOTHING", self.PARKOUR_ACTION_SMOOTHING)
        )
        self.parkour_debug = os.environ.get("PARKOUR_DEBUG", "1").lower() not in {"0", "false", "no"}
        self.parkour_joint_order = os.environ.get("PARKOUR_JOINT_ORDER", "env").lower()
        self.parkour_scan_mode = os.environ.get("PARKOUR_SCAN_MODE", "env").lower()
        self.parkour_scan_flat_value = float(os.environ.get("PARKOUR_SCAN_FLAT_VALUE", "0.0"))
        self.parkour_joint_pos_mode = os.environ.get("PARKOUR_JOINT_POS_MODE", "env").lower()
        self.parkour_command_mode = os.environ.get("PARKOUR_COMMAND_MODE", "vx_only").lower()
        self.parkour_prop_mode = os.environ.get("PARKOUR_PROP_MODE", "atec").lower()

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
        if self.parkour_scan_mode not in {"env", "zero", "flat"}:
            raise ValueError("PARKOUR_SCAN_MODE must be one of: env, zero, flat")
        if self.parkour_joint_pos_mode not in {"env", "zero"}:
            raise ValueError("PARKOUR_JOINT_POS_MODE must be one of: env, zero")
        if self.parkour_command_mode not in {"vx_only", "xyz"}:
            raise ValueError("PARKOUR_COMMAND_MODE must be one of: vx_only, xyz")
        if self.parkour_prop_mode not in {"atec", "locomotion_scaled", "parkour48"}:
            raise ValueError("PARKOUR_PROP_MODE must be one of: atec, locomotion_scaled, parkour48")
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

        if self.parkour_debug:
            print(
                "[PARKOUR-MODEL] "
                f"prop={self.parkour_prop_dim} scan={self.parkour_scan_dim} "
                f"priv_explicit={self.parkour_priv_explicit_dim} "
                f"priv_latent={self.parkour_priv_latent_dim} hist={self.parkour_history_len} "
                f"total={self.parkour_total_obs_dim} actions={self.leg_action_dim}"
            )

    def _policy_int_attr(self, name: str, default: int) -> int:
        actor = getattr(self.policy, "actor", None)
        source = actor if actor is not None else self.policy
        try:
            value = getattr(source, name)
        except Exception:
            return int(default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

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
        vx = max(self.parkour_min_vx, min(self.parkour_max_vx, vx))
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
        if self.parkour_joint_pos_mode == "zero":
            joint_pos_leg = torch.zeros_like(joint_pos_leg)
        scale = self.train_to_env_action_scale.to(dtype=proprio.dtype)
        last_action_leg = actions_policy_leg / scale

        cmd = self._get_velocity_commands(proprio)
        if self.parkour_prop_mode == "atec":
            prop = torch.cat(
                [
                    base_lin_vel,
                    base_ang_vel,
                    cmd,
                    projected_gravity,
                    joint_pos_leg,
                    joint_vel_leg,
                    actions_policy_leg,
                ],
                dim=-1,
            )
        elif self.parkour_prop_mode == "locomotion_scaled":
            prop = torch.cat(
                [
                    base_lin_vel * 2.0,
                    base_ang_vel * 0.25,
                    cmd,
                    projected_gravity,
                    joint_pos_leg,
                    joint_vel_leg * 0.05,
                    last_action_leg,
                ],
                dim=-1,
            )
        else:
            imu_roll_pitch = projected_gravity[:, :2]
            zeros_1 = torch.zeros((num_envs, 1), device=self.device, dtype=proprio.dtype)
            zero_cmd_xy = torch.zeros((num_envs, 2), device=self.device, dtype=proprio.dtype)
            env_non_flat = torch.ones((num_envs, 1), device=self.device, dtype=proprio.dtype)
            if self.parkour_command_mode == "xyz":
                cmd_xy = cmd[:, 1:3]
                cmd_vx = cmd[:, 0:1]
            else:
                cmd_xy = zero_cmd_xy
                cmd_vx = cmd[:, 0:1]
            prop = torch.cat(
                [
                    base_ang_vel * 0.25,
                    imu_roll_pitch,
                    zeros_1,
                    zeros_1,
                    zeros_1,
                    cmd_xy,
                    cmd_vx,
                    env_non_flat,
                    joint_pos_leg,
                    joint_vel_leg * 0.05,
                    last_action_leg,
                ],
                dim=-1,
            )
        if prop.shape[-1] != self.parkour_prop_dim:
            raise RuntimeError(f"Parkour prop dim mismatch: {prop.shape[-1]}")

        extero = obs.get("extero")
        if self.parkour_scan_mode == "zero":
            scan = torch.zeros(
                (num_envs, self.parkour_scan_dim), device=self.device, dtype=proprio.dtype
            )
        elif self.parkour_scan_mode == "flat":
            scan = torch.full(
                (num_envs, self.parkour_scan_dim),
                self.parkour_scan_flat_value,
                device=self.device,
                dtype=proprio.dtype,
            )
        elif extero is None:
            scan = torch.zeros(
                (num_envs, self.parkour_scan_dim), device=self.device, dtype=proprio.dtype
            )
        else:
            scan = extero.to(device=self.device, dtype=proprio.dtype)
            if scan.ndim == 1:
                scan = scan.view(1, -1)
            else:
                scan = scan.reshape(scan.shape[0], -1)
            if scan.shape[-1] < self.parkour_scan_dim:
                pad = torch.zeros(
                    (scan.shape[0], self.parkour_scan_dim - scan.shape[-1]),
                    device=self.device,
                    dtype=proprio.dtype,
                )
                scan = torch.cat([scan, pad], dim=-1)
            elif scan.shape[-1] > self.parkour_scan_dim:
                scan = scan[:, :self.parkour_scan_dim]
            scan = torch.nan_to_num(scan, nan=0.0, posinf=1.0, neginf=-1.0)
            scan = torch.clamp(scan, -1.0, 1.0)

        if self.parkour_priv_explicit_dim != 0 or self.parkour_priv_latent_dim != 0 or self.parkour_history_len != 0:
            raise RuntimeError(
                "This adapter currently supports exported parkour policies with "
                "num_priv_explicit=0, num_priv_latent=0, and num_hist=0."
            )

        policy_obs = torch.cat(
            [
                prop,
                scan,
            ],
            dim=-1,
        )
        if policy_obs.shape[-1] != self.parkour_total_obs_dim:
            raise RuntimeError(f"Parkour obs dim mismatch: {policy_obs.shape[-1]}")
        if self.parkour_debug and not self._printed_policy_debug:
            print(
                "[PARKOUR-OBS] "
                f"proprio={tuple(proprio.shape)} policy_obs={tuple(policy_obs.shape)} "
                f"cmd=({cmd[0,0].item():+.2f},{cmd[0,1].item():+.2f},{cmd[0,2].item():+.2f}) "
                f"prop=[{prop.min().item():+.2f},{prop.max().item():+.2f}] "
                f"joint_pos=[{joint_pos_leg.min().item():+.2f},{joint_pos_leg.max().item():+.2f}] "
                f"joint_vel=[{joint_vel_leg.min().item():+.2f},{joint_vel_leg.max().item():+.2f}] "
                f"last_action=[{last_action_leg.min().item():+.2f},{last_action_leg.max().item():+.2f}] "
                f"scan=[{scan.min().item():+.2f},{scan.max().item():+.2f}] "
                f"scale={scale[0,0].item():.3f} "
                f"joint_order={self.parkour_joint_order} "
                f"scan_mode={self.parkour_scan_mode} "
                f"joint_pos_mode={self.parkour_joint_pos_mode} "
                f"command_mode={self.parkour_command_mode} "
                f"prop_mode={self.parkour_prop_mode}"
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
            start = max(0.0, min(1.0, self.parkour_start_factor))
            progress = min(1.0, max(0.0, float(self.step) / float(self.parkour_ramp_steps)))
            ramp = start + (1.0 - start) * progress
            action_env[:, self.leg_joint_indices] *= ramp

        if self._last_action_env is None or self._last_action_env.shape != action_env.shape:
            self._last_action_env = action_env.detach()
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
            action_values = [round(float(v), 2) for v in action_train[0].detach().cpu().tolist()]
            print(
                "[PARKOUR-ACT] "
                f"train=[{action_train.min().item():+.2f},{action_train.max().item():+.2f}] "
                f"mean={action_train.mean().item():+.2f} "
                f"values={action_values}"
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
