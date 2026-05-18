from __future__ import annotations

from collections import deque
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from solution_act import Agent, Args


class AlgSolution:
    """Submission entrypoint for Task E using the ACT baseline policy."""

    _QPOS_SLICE = slice(0, 8)
    _QVEL_SLICE = slice(8, 16)

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ckpt_path = self._resolve_checkpoint_path()
        ckpt = torch.load(ckpt_path, map_location=self.device)
        norm_stats = ckpt["norm_stats"]
        state_dim = norm_stats["state_mean"].shape[-1]
        act_dim = norm_stats["action_mean"].shape[-1]
        weight_key = "ema_agent" if "ema_agent" in ckpt else "agent"

        train_args = Args()
        train_args.num_queries = 30
        train_args.include_rgb = any("backbone" in key for key in ckpt[weight_key].keys())

        self.agent = Agent(state_dim, act_dim, train_args).to(self.device)
        self.agent.load_state_dict(ckpt[weight_key])
        self.agent.eval()

        self.num_queries = train_args.num_queries
        self.temporal_agg = True
        self._temporal_decay = 0.01

        self.state_mean = norm_stats["state_mean"].to(self.device)
        self.state_std = norm_stats["state_std"].to(self.device)
        self.act_mean = norm_stats["action_mean"].to(self.device)
        self.act_std = norm_stats["action_std"].to(self.device)

        self.default_joint_pos = torch.tensor(
            [[0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035]],
            dtype=torch.float32,
            device=self.device,
        )
        self.teleop_home_joint_pos = torch.tensor(
            [[-0.000033, 0.924525, -1.514983, 0.000011, 1.219900, -0.000033, 0.035000, -0.035000]],
            dtype=torch.float32,
            device=self.device,
        )

        self._startup_zero_steps = 25
        self._home_qpos_tolerance = 0.10
        self._home_kp = 2.0
        self._home_kd = 0.2

        self._startup_step = 0
        self._home_done = False
        self._ts = 0
        self._action_history: deque[torch.Tensor] = deque(maxlen=self.num_queries)
        self._last_action_seq: torch.Tensor | None = None

    def _resolve_checkpoint_path(self) -> Path:
        demo_dir = Path(__file__).resolve().parent
        repo_root = demo_dir.parent
        candidates = [
            demo_dir / "policy_act.pt",
            repo_root / "atec_robot_model" / "baseline" / "act" / "policy.pt",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            "ACT checkpoint not found. Checked: "
            + ", ".join(str(path) for path in candidates)
        )

    def _compute_home_action(self, proprio: torch.Tensor) -> tuple[torch.Tensor, bool]:
        joint_pos_rel = proprio[:, self._QPOS_SLICE]
        joint_vel_rel = proprio[:, self._QVEL_SLICE]
        qpos = joint_pos_rel + self.default_joint_pos
        qerr = self.teleop_home_joint_pos - qpos

        within_tolerance = torch.all(torch.abs(qerr) <= self._home_qpos_tolerance, dim=1)
        action = torch.clamp((self._home_kp * qerr - self._home_kd * joint_vel_rel) / 0.5, -1.0, 1.0)
        if bool(torch.all(within_tolerance)):
            action = torch.zeros_like(action)
        return action, bool(torch.all(within_tolerance))

    def _build_model_obs(self, obs: dict, proprio: torch.Tensor) -> dict:
        joint_pos_rel = proprio[:, self._QPOS_SLICE]
        qpos = joint_pos_rel + self.default_joint_pos
        model_obs = {"state": (qpos - self.state_mean) / self.state_std}

        if self.agent.include_rgb:
            rgb = obs["image"]["video_rgb"].to(self.device)
            if rgb.ndim == 4 and rgb.shape[1] == 4:
                rgb = rgb[:, :3]
            if rgb.ndim == 4 and rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            if rgb.dtype != torch.uint8:
                rgb = (rgb.float() * 255.0).clamp(0, 255).to(torch.uint8)
            if not (rgb.ndim == 4 and rgb.shape[1] in (3, 4)):
                rgb = rgb.permute(0, 3, 1, 2)
            if rgb.shape[-2:] != (224, 224):
                rgb = TF.resize(
                    rgb,
                    [224, 224],
                    interpolation=TF.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            model_obs["rgb"] = rgb.unsqueeze(1)

        return model_obs

    def predicts(self, obs, current_score):
        if not isinstance(obs, dict) or "proprio" not in obs:
            raise ValueError("Expected observation dict with a 'proprio' entry.")

        proprio = obs["proprio"].to(self.device)

        if self._startup_step < self._startup_zero_steps:
            self._startup_step += 1
            action = torch.zeros((proprio.shape[0], self.agent.act_dim), device=self.device)
            return {"action": action.cpu().tolist(), "giveup": False}

        if not self._home_done:
            action, home_reached = self._compute_home_action(proprio)
            if home_reached:
                self._home_done = True
                self._ts = 0
                self._action_history.clear()
                self._last_action_seq = None
            return {"action": action.cpu().tolist(), "giveup": False}

        model_obs = self._build_model_obs(obs, proprio)
        ts = self._ts
        query_frequency = 1 if self.temporal_agg else self.num_queries

        if ts % query_frequency == 0:
            with torch.no_grad():
                action_seq = self.agent.get_action(model_obs)
            if self.temporal_agg:
                self._action_history.append(action_seq)
            else:
                self._last_action_seq = action_seq

        if self.temporal_agg:
            num_entries = len(self._action_history)
            actions_for_current = torch.stack(
                [seq[:, num_entries - 1 - idx, :] for idx, seq in enumerate(self._action_history)],
                dim=1,
            )
            weights = torch.exp(-self._temporal_decay * torch.arange(num_entries, device=self.device))
            weights = (weights / weights.sum()).unsqueeze(0).unsqueeze(-1)
            raw_action = (actions_for_current * weights).sum(dim=1)
        else:
            raw_action = self._last_action_seq[:, ts % query_frequency]

        action = raw_action * self.act_std + self.act_mean
        self._ts += 1
        return {"action": action.cpu().tolist(), "giveup": False}
