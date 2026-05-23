"""
Task D: Push box into pit, then cross.

Scoring (36 total):
  - Robot crossing x=-1.4  → +2
  - Robot crossing x=2.0  → +20
  - Box in pit [-0.7, 0.7] → +14
  - Success: robot x > 3.5

Layout:
  - Robot at (-3, 0), box at (-3, 1.6), pit at x≈0
  - Box must be pushed into pit (x-range [-0.7, 0.7])
  - Robot must then cross the pit

Sequence:
  1. APPROACH  → Walk forward toward box's X, then strafe to align
  2. PUSH_BOX  → Push box toward pit (x direction)
  3. CROSS     → Walk across pit

The robot can ONLY walk forward (vel_x direction). It must approach
from the side of the box and push forward.
"""

import os
import math
import torch


class AlgSolution:

    def __init__(self):
        policy_path = os.path.dirname(os.path.abspath(__file__)) + '/policy.pt'
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.arm_action_dim = 8

        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))

        self.train_to_env_action_scale = torch.tensor(
            [0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5, 0.25, 0.5, 0.5],
            device=self.device, dtype=torch.float32,
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0],
            device=self.device, dtype=torch.float32,
        ).view(1, -1)

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim), device=self.device, dtype=torch.float32,
        )

        # ── Velocity command ─────────────────────────────────────────────────
        # vel_x = forward speed, vel_y = heading (radians), vel_z = turn rate
        self._vel_x = 0.5
        self._vel_y = 0.0  # heading (0 = face +X)
        self._vel_z = 0.0

                # ── State machine ────────────────────────────────────────────────────────
        # Robot at (-3, 0), box at (-3, 1.6), pit at x≈0
        # Robot faces +X (forward). Box is to the LEFT (+Y) of robot.
        # vel_y = heading in radians: 0 = face +X, +1.57 = face +Y (left)
        self.phase = "TO_SIDE"
        self.step = 0

        # ── LiDAR box tracking ─────────────────────────────────────────────────
        self.lidar_box = None

        # ── Step limits ────────────────────────────────────────────────────────
        self.TO_SIDE_STEPS = 400     # walk to +Y side of box
        self.PUSH_STEPS = 800        # push box toward pit
        self.CROSS_STEPS = 400       # walk across pit

        # ── Diagnostic ─────────────────────────────────────────────────────────
        self._last_phase = None
        self._printed_obs = False

    # ══════════════════════════════════════════════════════════════════════════
    # LiDAR processing
    # ══════════════════════════════════════════════════════════════════════════

    def _get_lidar_scan(self, obs) -> torch.Tensor | None:
        extero = obs.get("extero")
        if extero is None or extero.numel() == 0:
            return None
        scan = extero.to(device=self.device, dtype=torch.float32)
        if scan.ndim == 1:
            scan = scan.view(1, -1)
        elif scan.ndim > 2:
            scan = scan.reshape(scan.shape[0], -1)
        return scan[0]

    def _detect_box_lidar(self, obs) -> dict | None:
        """Detect box cluster in LiDAR scan."""
        scan = self._get_lidar_scan(obs)
        if scan is None or scan.numel() < 32:
            return None

        flat = scan.flatten()
        finite_mask = flat.isfinite()
        values = flat[finite_mask]
        if values.numel() < 16:
            return None

        n = flat.numel()
        if n % 360 == 0:
            cols = flat.view(-1, 360)
            col_finite = cols.isfinite()
            safe = torch.where(col_finite, cols, torch.zeros_like(cols))
            counts = col_finite.sum(dim=0).clamp_min(1)
            horizontal = safe.sum(dim=0) / counts
        else:
            horizontal = flat

        n_bins = horizontal.numel()
        median = values.median()
        deviation = (horizontal - median).abs()
        valid_dev = deviation[horizontal.isfinite()]
        if valid_dev.numel() < 8:
            return None

        kth = max(1, int(valid_dev.numel() * 0.88))
        threshold = valid_dev.kthvalue(kth).values.clamp_min(0.06)
        mask = horizontal.isfinite() & (deviation >= threshold)

        indices = torch.where(mask)[0].cpu().tolist()
        if not indices:
            return None

        clusters = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                clusters.append((start, prev))
                start = prev = idx
        clusters.append((start, prev))

        if (len(clusters) > 1 and clusters[0][0] == 0
                and clusters[-1][1] == n_bins - 1):
            last = clusters.pop(-1)
            first = clusters.pop(0)
            clusters.insert(0, (last[0], first[1] + n_bins))

        best = None
        best_score = -1.0

        for s, e in clusters:
            width = e - s + 1
            angular_w = float(width) * (2 * math.pi / float(n_bins))
            if width < 4 or angular_w < 0.08 or angular_w > 1.2:
                continue

            est_range = 1.0 / max(angular_w, 0.01)
            est_range = max(0.4, min(6.0, est_range))

            idxs = torch.arange(s, e + 1, device=self.device) % n_bins
            angles = (idxs.float() / float(n_bins - 1)) * (2 * math.pi) - math.pi
            weights = deviation[idxs].clamp_min(1e-4)
            sin_mean = (weights * torch.sin(angles)).sum() / weights.sum()
            cos_mean = (weights * torch.cos(angles)).sum() / weights.sum()
            bearing = math.atan2(sin_mean.item(), cos_mean.item())

            range_score = 1.0 / (1.0 + 0.5 * abs(est_range - 1.5))
            bearing_score = 1.0 / (1.0 + 0.3 * abs(bearing))
            width_score = math.sqrt(float(width))
            score = width_score * range_score * bearing_score

            if score > best_score:
                best_score = score
                best = (bearing, est_range, angular_w, width)

        if best is None:
            return None

        bearing, est_range, angular_w, width = best
        return {
            "bearing": bearing,
            "range": est_range,
            "angular_width": angular_w,
            "count": width,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # State machine
    # ══════════════════════════════════════════════════════════════════════════

    def _transition(self) -> None:
        p = self.phase
        s = self.step
        lb = self.lidar_box

        if p == "TO_SIDE":
            # Strafe left (heading=+1.57) toward box's Y. Transition when box is close.
            if lb is not None and lb["range"] < 1.0:
                self.phase = "PUSH_TO_PIT"
                self.step = 0
            elif s >= self.TO_SIDE_STEPS:
                self.phase = "PUSH_TO_PIT"
                self.step = 0

        elif p == "PUSH_TO_PIT":
            # Face +X (heading=0) and push box toward pit
            if s >= self.PUSH_STEPS:
                self.phase = "CROSS"
                self.step = 0

        elif p == "CROSS":
            pass

    # ══════════════════════════════════════════════════════════════════════════
    # Policy interface (mirrors solution_rl.py)
    # ══════════════════════════════════════════════════════════════════════════

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        num_envs = int(proprio.shape[0])
        cmd = torch.tensor(
            [self._vel_x, self._vel_y, self._vel_z],
            device=self.device, dtype=torch.float32,
        ).view(1, 3)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

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
        velocity_commands = self._get_velocity_commands(proprio)

        return torch.cat([
            base_ang_vel * 0.25,
            projected_gravity,
            velocity_commands,
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_train_leg,
        ], dim=-1)

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(f"Expected {self.leg_action_dim}, got {action_train.shape[-1]}")
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
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

    # ══════════════════════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════════════════════

    def predicts(self, obs, current_score):
        if not self._printed_obs:
            print("OBS KEYS:", list(obs.keys()))
            self._printed_obs = True

        if current_score > 1:
            return {'action': [], 'giveup': True}

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        # ── LiDAR sensing ─────────────────────────────────────────────────────
        lb = self._detect_box_lidar(obs)
        self.lidar_box = lb
        self._transition()

        p = self.phase

        # ── Velocity command ───────────────────────────────────────────────────
        # vel_x = forward speed (always 0.5)
        # vel_y = heading: 0=face +X, +1.57=face +Y (left)
        if p == "TO_SIDE":
            # Strafe toward box: heading = +1.57 (face +Y = left)
            self._vel_x = 0.3
            self._vel_y = 1.57  # face +Y (left)
            self._vel_z = 0.0
        elif p == "PUSH_TO_PIT":
            # Face +X and push forward
            self._vel_x = 0.5
            self._vel_y = 0.0   # face +X (forward)
            self._vel_z = 0.0
        elif p == "CROSS":
            self._vel_x = 0.4
            self._vel_y = 0.0
            self._vel_z = 0.0

        action = self._run_policy(obs, action_dim)

        # ── Log ──────────────────────────────────────────────────────────────
        if p != self._last_phase:
            lb_str = (f"rng={lb['range']:.2f} aw={lb['angular_width']:.2f}"
                     if lb else "none")
            print(
                f"[D] phase={p:<12} step={self.step:>3}  "
                f"lidar=[{lb_str}]  "
                f"cmd=(fwd={self._vel_x:+.1f}, hdg={self._vel_y:+.2f})"
            )
            self._last_phase = p

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}