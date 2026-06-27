"""Task D B2Piper controller: teleop/replay through a locomotion policy.

The stable locomotion controller is demo/policy.pt.official when available.
demo/policy.pt is treated as an experimental parkour policy: it is scored and
logged, but it is not allowed to drive the robot unless its first action looks
numerically healthy and PARKOUR_BLEND is explicitly raised above zero.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch


class AlgSolution:
    DEFAULT_REPLAY_NAME = "task_d_manual_best.json"
    REPLAY_END_STOP_STEPS = 50
    REPLAY_END_SLOW_FORWARD = 0.25

    LEG_ACTION_DIM = 12
    ARM_ACTION_DIM = 8
    LEG_JOINT_INDICES = list(range(12))
    ARM_JOINT_INDICES = list(range(12, 20))

    # Policy action -> TaskD raw env action. TaskD then applies scale=0.5.
    # This reproduces train joint deltas of hip=0.125 and thigh/calf=0.25.
    TRAIN_TO_ENV_SCALE = [0.25, 0.5, 0.5] * 4
    ENV_TO_TRAIN_SCALE = [4.0, 2.0, 2.0] * 4
    RAW_POLICY_CLIP = [1.2, 3.0, 3.0] * 4
    FINAL_ENV_CLIP = [0.35, 1.25, 1.25] * 4
    OFFICIAL_FINAL_ENV_CLIP = [0.5, 1.8, 1.8] * 4
    PARKOUR_RAW_ACTION_LIMIT = 4.0
    PARKOUR_PROFILE_SCORE_LIMIT = 12.0

    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.policy = self._load_policy(base_dir / "policy.pt", required=True)
        self.official_policy = self._load_policy(base_dir / "policy.pt.official", required=False)
        self.active_policy = "official" if self.official_policy is not None else "parkour"

        self.parkour_prop_dim = self._policy_int_attr(self.policy, "num_prop", 48)
        self.parkour_scan_dim = self._policy_int_attr(self.policy, "num_scan", 5760)
        self.parkour_priv_explicit_dim = self._policy_int_attr(self.policy, "num_priv_explicit", 0)
        self.parkour_priv_latent_dim = self._policy_int_attr(self.policy, "num_priv_latent", 0)
        self.parkour_history_len = self._policy_int_attr(self.policy, "num_hist", 0)
        self.leg_action_dim = self._policy_int_attr(self.policy, "num_actions", self.LEG_ACTION_DIM)
        self.parkour_total_obs_dim = (
            self.parkour_prop_dim
            + self.parkour_scan_dim
            + self.parkour_priv_explicit_dim
            + self.parkour_priv_latent_dim
            + self.parkour_history_len * self.parkour_prop_dim
        )

        self.parkour_debug = os.environ.get("PARKOUR_DEBUG", "1").lower() not in {"0", "false", "no"}
        self.parkour_profile = os.environ.get("PARKOUR_PROFILE", "auto").lower()
        self.parkour_scan_mode = os.environ.get("PARKOUR_SCAN_MODE", "zero").lower()
        self.parkour_scan_flat_value = float(os.environ.get("PARKOUR_SCAN_FLAT_VALUE", "0.0"))
        self.parkour_blend = max(0.0, min(1.0, float(os.environ.get("PARKOUR_BLEND", "0.0"))))
        self.parkour_action_smoothing = float(os.environ.get("PARKOUR_ACTION_SMOOTHING", "0.80"))
        self.parkour_ramp_steps = int(os.environ.get("PARKOUR_RAMP_STEPS", "75"))
        self.parkour_idle_vx = float(os.environ.get("PARKOUR_IDLE_VX", "0.0"))
        self.parkour_min_vx = float(os.environ.get("PARKOUR_MIN_VX", "-1.0"))
        self.parkour_max_vx = float(os.environ.get("PARKOUR_MAX_VX", "1.0"))
        self.yaw_hold_enabled = os.environ.get("PARKOUR_YAW_HOLD", "1").lower() not in {"0", "false", "no"}

        joint_order = os.environ.get("PARKOUR_JOINT_ORDER", "env").lower()
        if joint_order in {"env", "atec", "fr_fl_rr_rl"}:
            leg_perm = list(range(self.leg_action_dim))
        elif joint_order in {"fl_fr_rl_rr", "parkour"}:
            leg_perm = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        else:
            raise ValueError("PARKOUR_JOINT_ORDER must be env or parkour")

        self.env_to_policy_leg_perm = torch.tensor(leg_perm, device=self.device, dtype=torch.long)
        self.policy_to_env_leg_perm = torch.tensor(leg_perm, device=self.device, dtype=torch.long)
        self.train_to_env_scale = torch.tensor(
            self.TRAIN_TO_ENV_SCALE, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.env_to_train_scale = torch.tensor(
            self.ENV_TO_TRAIN_SCALE, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.raw_policy_clip = torch.tensor(
            self.RAW_POLICY_CLIP, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.final_env_clip = torch.tensor(
            self.FINAL_ENV_CLIP, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.official_final_env_clip = torch.tensor(
            self.OFFICIAL_FINAL_ENV_CLIP, device=self.device, dtype=torch.float32
        ).view(1, -1)
        self.arm_default_action = torch.zeros((1, self.ARM_ACTION_DIM), device=self.device)

        self._dt = 0.02
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_z = 0.0
        self.step = 0
        self.phase = "REPLAY"

        self.manual_control = False
        self.manual_world_frame = False
        self.manual_vx = 0.0
        self.manual_vy = 0.0
        self.manual_wz = 0.0

        self.replay_commands: list[dict[str, Any]] = []
        self.replay_index = 0
        self.replay_end_index: int | None = None
        self._replay_warned = False

        self._selected_profile: str | None = None
        self._parkour_enabled = self.official_policy is None
        self._parkour_disable_reason: str | None = None
        self._last_safe_action_env: torch.Tensor | None = None
        self._last_output_action_env: torch.Tensor | None = None
        self._unsafe_steps = 0
        self._printed_obs_keys = False
        self._printed_policy_action = False
        self._printed_env_action = False
        self._switched_to_official = False

        self._load_default_replay(base_dir / self.DEFAULT_REPLAY_NAME)
        if self.parkour_debug:
            print(
                "[PARKOUR-MODEL] "
                f"prop={self.parkour_prop_dim} scan={self.parkour_scan_dim} "
                f"priv_explicit={self.parkour_priv_explicit_dim} "
                f"priv_latent={self.parkour_priv_latent_dim} hist={self.parkour_history_len} "
                f"total={self.parkour_total_obs_dim} actions={self.leg_action_dim} "
                f"fallback={'yes' if self.official_policy is not None else 'no'}"
            )
            print(
                "[POLICY-MODE] "
                f"primary={self.active_policy} parkour_blend={self.parkour_blend:.2f} "
                f"parkour_gate_score<{self.PARKOUR_PROFILE_SCORE_LIMIT:.1f} "
                f"parkour_gate_raw<{self.PARKOUR_RAW_ACTION_LIMIT:.1f}"
            )

    def _load_policy(self, path: Path, required: bool):
        if not path.exists():
            if required:
                raise FileNotFoundError(path)
            return None
        policy = torch.jit.load(str(path), map_location=self.device)
        policy.eval()
        return policy

    @staticmethod
    def _policy_int_attr(policy, name: str, default: int) -> int:
        actor = getattr(policy, "actor", None)
        source = actor if actor is not None else policy
        try:
            return int(getattr(source, name))
        except Exception:
            return int(default)

    # ------------------------------------------------------------------
    # Public hooks used by scripts/play_atec_task.py
    # ------------------------------------------------------------------

    def set_manual_command(self, vx: float, vy: float, wz: float, world_frame: bool = False) -> None:
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
    # Pose and command helpers
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
        self.est_yaw = (self.est_yaw + math.pi) % (2.0 * math.pi) - math.pi

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

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        num_envs = int(proprio.shape[0])
        vx = self._vel_x
        vy = self._vel_y
        wz = self._vel_z

        if abs(vx) < 1.0e-6 and abs(vy) < 1.0e-6 and abs(wz) < 1.0e-6:
            vx = self.parkour_idle_vx

        if self.yaw_hold_enabled and abs(wz) < 0.05:
            yaw_rate = float(proprio[0, 5].item())
            wz = max(-1.0, min(1.0, -1.2 * self.est_yaw - 0.25 * yaw_rate))

        vx = max(self.parkour_min_vx, min(self.parkour_max_vx, vx))
        cmd = torch.tensor([vx, vy, wz], device=self.device, dtype=torch.float32).view(1, 3)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd.to(dtype=proprio.dtype)

    # ------------------------------------------------------------------
    # Observation adapter
    # ------------------------------------------------------------------

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

    def _leg_terms(self, joint_pos_all, joint_vel_all, actions_all):
        joint_pos_env = joint_pos_all[:, self.LEG_JOINT_INDICES]
        joint_vel_env = joint_vel_all[:, self.LEG_JOINT_INDICES]
        actions_env = actions_all[:, self.LEG_JOINT_INDICES]
        q = joint_pos_env[:, self.env_to_policy_leg_perm]
        qd = joint_vel_env[:, self.env_to_policy_leg_perm]
        last_env = actions_env[:, self.env_to_policy_leg_perm]
        last_train = last_env * self.env_to_train_scale.to(dtype=actions_all.dtype)
        return q, qd, last_env, last_train

    def _scan_tensor(self, obs, num_envs: int, dtype: torch.dtype) -> torch.Tensor:
        if self.parkour_scan_mode == "zero":
            return torch.zeros((num_envs, self.parkour_scan_dim), device=self.device, dtype=dtype)
        if self.parkour_scan_mode == "flat":
            return torch.full(
                (num_envs, self.parkour_scan_dim),
                self.parkour_scan_flat_value,
                device=self.device,
                dtype=dtype,
            )

        extero = obs.get("extero")
        if extero is None:
            return torch.zeros((num_envs, self.parkour_scan_dim), device=self.device, dtype=dtype)
        scan = extero.to(device=self.device, dtype=dtype)
        if scan.ndim == 1:
            scan = scan.view(1, -1)
        else:
            scan = scan.reshape(scan.shape[0], -1)
        if scan.shape[-1] < self.parkour_scan_dim:
            pad = torch.zeros(
                (scan.shape[0], self.parkour_scan_dim - scan.shape[-1]),
                device=self.device,
                dtype=dtype,
            )
            scan = torch.cat([scan, pad], dim=-1)
        elif scan.shape[-1] > self.parkour_scan_dim:
            scan = scan[:, :self.parkour_scan_dim]
        scan = torch.nan_to_num(scan, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(scan, -1.0, 1.0)

    def _candidate_props(self, obs, action_dim: int) -> dict[str, torch.Tensor]:
        (
            proprio,
            base_lin,
            base_ang,
            gravity,
            joint_pos_all,
            joint_vel_all,
            actions_all,
        ) = self._split_proprio(obs, action_dim)
        q, qd, last_env, last_train = self._leg_terms(joint_pos_all, joint_vel_all, actions_all)
        cmd = self._get_velocity_commands(proprio)

        return {
            "taskd_raw": torch.cat([base_lin, base_ang, cmd, gravity, q, qd, last_env], dim=-1),
            "taskd_train_action": torch.cat([base_lin, base_ang, cmd, gravity, q, qd, last_train], dim=-1),
            "rough_locomotion": torch.cat([base_lin, base_ang, gravity, cmd, q, qd, last_train], dim=-1),
            "scaled_locomotion": torch.cat(
                [base_lin * 2.0, base_ang * 0.25, gravity, cmd, q, qd * 0.05, last_train],
                dim=-1,
            ),
        }

    def _policy_obs_from_prop(self, obs, prop: torch.Tensor) -> torch.Tensor:
        if prop.shape[-1] != self.parkour_prop_dim:
            raise RuntimeError(f"Parkour prop dim mismatch: {prop.shape[-1]} != {self.parkour_prop_dim}")
        if self.parkour_priv_explicit_dim or self.parkour_priv_latent_dim or self.parkour_history_len:
            raise RuntimeError("Unsupported parkour export: priv/history dims are non-zero.")
        scan = self._scan_tensor(obs, int(prop.shape[0]), prop.dtype)
        policy_obs = torch.cat([prop, scan], dim=-1)
        if policy_obs.shape[-1] != self.parkour_total_obs_dim:
            raise RuntimeError(
                f"Parkour obs dim mismatch: {policy_obs.shape[-1]} != {self.parkour_total_obs_dim}"
            )
        return policy_obs

    def _score_action(self, action: torch.Tensor) -> float:
        if not torch.isfinite(action).all():
            return float("inf")
        abs_action = action.abs()
        max_abs = float(abs_action.max().item())
        if max_abs > self.PARKOUR_RAW_ACTION_LIMIT:
            return float("inf")
        hip_mean = float(abs_action[:, 0::3].mean().item())
        saturation_count = float((abs_action > self.raw_policy_clip.to(action.dtype)).sum().item())
        if saturation_count > 0:
            return float("inf")
        return float(abs_action.mean().item()) + 0.5 * max_abs + 2.0 * hip_mean + 5.0 * saturation_count

    def _select_parkour_profile(self, obs, action_dim: int) -> str:
        if self._parkour_disable_reason is not None:
            raise RuntimeError(f"Parkour policy disabled: {self._parkour_disable_reason}")

        candidates = self._candidate_props(obs, action_dim)
        valid_names = set(candidates)
        if self.parkour_profile != "auto":
            if self.parkour_profile not in valid_names:
                raise ValueError(f"PARKOUR_PROFILE must be auto or one of {sorted(valid_names)}")
            self._selected_profile = self.parkour_profile
            self._parkour_enabled = True
            return self._selected_profile

        scores: dict[str, float] = {}
        with torch.inference_mode():
            for name, prop in candidates.items():
                policy_obs = self._policy_obs_from_prop(obs, prop)
                action = self.policy(policy_obs)
                if not isinstance(action, torch.Tensor):
                    action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
                if action.ndim == 1:
                    action = action.unsqueeze(0)
                scores[name] = self._score_action(action.to(device=self.device, dtype=torch.float32))

        self._selected_profile = min(scores, key=scores.get)
        best_score = scores[self._selected_profile]
        if self.parkour_debug:
            formatted = ", ".join(f"{k}:{v:.2f}" if math.isfinite(v) else f"{k}:inf" for k, v in scores.items())
            print(f"[PARKOUR-PROFILE] selected={self._selected_profile} scores={formatted}")
        if best_score >= self.PARKOUR_PROFILE_SCORE_LIMIT or not math.isfinite(best_score):
            self._parkour_enabled = False
            self._parkour_disable_reason = "profile_score"
            if self.parkour_debug:
                formatted = ", ".join(f"{k}:{v:.2f}" if math.isfinite(v) else f"{k}:inf" for k, v in scores.items())
                print(
                    "[PARKOUR-DISABLED] "
                    f"reason=profile_score best={self._selected_profile}:{best_score:.2f} "
                    f"scores={formatted}"
                )
            raise RuntimeError("Parkour policy disabled by profile score gate")
        self._parkour_enabled = True
        return self._selected_profile

    def _extract_parkour_obs(self, obs, action_dim: int) -> torch.Tensor:
        profile = self._selected_profile or self._select_parkour_profile(obs, action_dim)
        props = self._candidate_props(obs, action_dim)
        prop = props[profile]
        return self._policy_obs_from_prop(obs, prop)

    def _extract_official_obs(self, obs, action_dim: int) -> torch.Tensor:
        (
            proprio,
            _base_lin,
            base_ang,
            gravity,
            joint_pos_all,
            joint_vel_all,
            actions_all,
        ) = self._split_proprio(obs, action_dim)
        q, qd, _last_env, last_train = self._leg_terms(joint_pos_all, joint_vel_all, actions_all)
        cmd = self._get_velocity_commands(proprio)
        return torch.cat([base_ang * 0.25, gravity, cmd, q, qd * 0.05, last_train], dim=-1)

    # ------------------------------------------------------------------
    # Policy and action mapping
    # ------------------------------------------------------------------

    def _policy_forward(self, policy, policy_obs: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            action = policy(policy_obs)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return action

    def _switch_to_official(self, reason: str) -> None:
        if self.active_policy == "official":
            return
        if self.official_policy is None:
            if self.parkour_debug and not self._switched_to_official:
                print(f"[POLICY-WARN] parkour unsafe ({reason}); official fallback missing")
                self._switched_to_official = True
            return
        self.active_policy = "official"
        self._last_output_action_env = None
        self._last_safe_action_env = None
        self._printed_policy_action = False
        self._printed_env_action = False
        print(f"[POLICY-FALLBACK] parkour unsafe; switched to official ({reason})")

    def _raw_action_is_unsafe(self, action_train: torch.Tensor) -> bool:
        if not torch.isfinite(action_train).all():
            return True
        return bool(action_train.abs().max().item() > self.PARKOUR_RAW_ACTION_LIMIT)

    def _map_official_action_to_env(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.LEG_ACTION_DIM:
            raise ValueError(f"Expected {self.LEG_ACTION_DIM}, got {action_train.shape[-1]}")

        scale = self.train_to_env_scale.to(dtype=action_train.dtype)
        final_clip = self.official_final_env_clip.to(dtype=action_train.dtype)
        leg_action_env = action_train[:, self.policy_to_env_leg_perm] * scale
        leg_action_env = torch.clamp(leg_action_env, -final_clip, final_clip)

        action_env = torch.zeros((action_train.shape[0], action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.LEG_JOINT_INDICES] = leg_action_env
        action_env[:, self.ARM_JOINT_INDICES] = self.arm_default_action.repeat(action_train.shape[0], 1)
        return action_env

    def _map_parkour_action_to_env(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.LEG_ACTION_DIM:
            raise ValueError(f"Expected {self.LEG_ACTION_DIM}, got {action_train.shape[-1]}")

        unsafe = self._raw_action_is_unsafe(action_train)
        if unsafe:
            self._unsafe_steps += 1
            if self._last_safe_action_env is not None:
                return self._last_safe_action_env.clone()
            raise RuntimeError("Unsafe parkour action before any safe action")
        else:
            self._unsafe_steps = 0

        raw_clip = self.raw_policy_clip.to(dtype=action_train.dtype)
        final_clip = self.final_env_clip.to(dtype=action_train.dtype)
        scale = self.train_to_env_scale.to(dtype=action_train.dtype)

        clipped_train = torch.clamp(action_train, -raw_clip, raw_clip)
        leg_action_env = clipped_train[:, self.policy_to_env_leg_perm] * scale
        leg_action_env = torch.clamp(leg_action_env, -final_clip, final_clip)

        if self.parkour_ramp_steps > 0:
            ramp = min(1.0, max(0.0, float(self.step) / float(self.parkour_ramp_steps)))
            leg_action_env = leg_action_env * ramp

        action_env = torch.zeros((action_train.shape[0], action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.LEG_JOINT_INDICES] = leg_action_env
        action_env[:, self.ARM_JOINT_INDICES] = self.arm_default_action.repeat(action_train.shape[0], 1)

        alpha = max(0.0, min(0.95, self.parkour_action_smoothing))
        if self._last_output_action_env is None or self._last_output_action_env.shape != action_env.shape:
            smoothed = action_env
        else:
            smoothed = alpha * self._last_output_action_env + (1.0 - alpha) * action_env
        self._last_output_action_env = smoothed.detach()
        if not unsafe:
            self._last_safe_action_env = smoothed.detach()
        return smoothed

    def _maybe_switch_for_spin(self, proprio: torch.Tensor) -> None:
        if self.active_policy != "parkour" or self.step < 10:
            return
        yaw_rate = abs(float(proprio[0, 5].item()))
        if abs(self.est_yaw) > 1.0 or yaw_rate > 4.0 or self._unsafe_steps >= 30:
            self._switch_to_official("spin_or_invalid_action")

    def _run_policy(self, obs, action_dim: int) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        self._maybe_switch_for_spin(proprio)

        official_obs = None
        official_action_train = None
        official_action_env = None
        if self.official_policy is not None:
            official_obs = self._extract_official_obs(obs, action_dim)
            official_action_train = self._policy_forward(self.official_policy, official_obs)
            official_action_env = self._map_official_action_to_env(official_action_train, action_dim)

        parkour_obs = None
        parkour_action_train = None
        parkour_action_env = None
        if self._parkour_disable_reason is None:
            try:
                parkour_obs = self._extract_parkour_obs(obs, action_dim)
                parkour_action_train = self._policy_forward(self.policy, parkour_obs)
                if self._raw_action_is_unsafe(parkour_action_train):
                    self._unsafe_steps += 1
                    self._parkour_enabled = False
                    self._parkour_disable_reason = "raw_action"
                    if self.parkour_debug:
                        print(
                            "[PARKOUR-DISABLED] "
                            f"reason=raw_action max_abs={parkour_action_train.abs().max().item():.2f}"
                        )
                else:
                    self._unsafe_steps = 0
                    self._parkour_enabled = True
                    if self.parkour_blend > 0.0 or self.official_policy is None:
                        parkour_action_env = self._map_parkour_action_to_env(parkour_action_train, action_dim)
            except Exception as exc:
                self._parkour_enabled = False
                self._parkour_disable_reason = str(exc)
                if self.parkour_debug:
                    print(f"[PARKOUR-DISABLED] reason={self._parkour_disable_reason}")

        if official_action_env is None:
            if parkour_action_env is None:
                raise RuntimeError("No usable policy action: official missing and parkour disabled")
            action_env = parkour_action_env
            action_train = parkour_action_train
            policy_obs = parkour_obs
            self.active_policy = "parkour"
        elif parkour_action_env is not None and self.parkour_blend > 0.0:
            blend = self.parkour_blend
            action_env = (1.0 - blend) * official_action_env + blend * parkour_action_env
            action_train = official_action_train
            policy_obs = official_obs
            self.active_policy = f"blend{blend:.2f}"
        else:
            action_env = official_action_env
            action_train = official_action_train
            policy_obs = official_obs
            self.active_policy = "official"

        if self.parkour_debug and not self._printed_policy_action:
            values = [round(float(v), 2) for v in action_train[0].detach().cpu().tolist()]
            parkour_status = "enabled" if self._parkour_enabled else f"disabled:{self._parkour_disable_reason}"
            print(
                f"[POLICY-ACT] mode={self.active_policy} profile={self._selected_profile or 'official'} "
                f"obs={tuple(policy_obs.shape)} train=[{action_train.min().item():+.2f},{action_train.max().item():+.2f}] "
                f"mean={action_train.mean().item():+.2f} parkour={parkour_status} values={values}"
            )
            self._printed_policy_action = True

        if self.parkour_debug and not self._printed_env_action:
            leg_env = action_env[:, self.LEG_JOINT_INDICES]
            hip = leg_env[:, 0::3]
            thigh = leg_env[:, 1::3]
            calf = leg_env[:, 2::3]
            print(
                "[POLICY-ENV] "
                f"final=[{leg_env.min().item():+.2f},{leg_env.max().item():+.2f}] "
                f"hip=[{hip.min().item():+.2f},{hip.max().item():+.2f}] "
                f"thigh=[{thigh.min().item():+.2f},{thigh.max().item():+.2f}] "
                f"calf=[{calf.min().item():+.2f},{calf.max().item():+.2f}] "
                f"official_clip=[{self.OFFICIAL_FINAL_ENV_CLIP[0]:.2f},"
                f"{self.OFFICIAL_FINAL_ENV_CLIP[1]:.2f},{self.OFFICIAL_FINAL_ENV_CLIP[2]:.2f}]"
            )
            self._printed_env_action = True

        return action_env

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def predicts(self, obs, current_score):
        if not self._printed_obs_keys:
            print("OBS KEYS:", list(obs.keys()))
            self._printed_obs_keys = True

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
                f"[D]{self.phase:<18}|{self.step:<4}|policy={self.active_policy:<8}|"
                f"cmd{frame}=({vx:+.2f},{vy:+.2f},{wz:+.2f})|"
                f"robot=({self.est_x:+.1f},{self.est_y:+.1f},{math.degrees(self.est_yaw):+.0f}deg)"
            )

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}
