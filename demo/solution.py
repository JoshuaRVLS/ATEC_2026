"""
Task D: Push box into pit, then cross.

World layout:
  - Robot spawns at (-3, 0), facing +X
  - Box at (-3, 1.6), pit at x≈0 (reward zone x ∈ [-0.7, 0.7])
  - Robot must maneuver around box and push it into pit

Sequence (coordinate-based):
  1. BACK      → back up from box for clearance
  2. LEFT      → walk to Y > box_Y (left side of box)
  3. PUSH_RIGHT→ push box in +X direction
  4. BACK_SIDE → back up to Y < box_Y (behind box)
  5. PUSH_PIT  → push rotated box into pit
  6. CROSS     → walk across pit

Transitions use actual robot WORLD POSITION (from dead reckoning).
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

        # ── Timing ────────────────────────────────────────────────────────
        self._dt = 0.02  # decimation=4, sim.dt=0.005

        # ── Robot pose (dead reckoning from base_lin_vel) ─────────────────
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0

        # ── Box target Y (from known init position) ────────────────────────
        self.BOX_Y = 1.6      # box's Y position
        self.BACK_X = -3.5    # back up to this X
        self.PIT_X = 1.0      # cross pit until this X

        # ── Velocity command ───────────────────────────────────────────────
        # Convention (from testing):
        #   vel_x = forward speed (+X world)
        #   vel_y = strafe: +value=LEFT, -value=RIGHT
        #   vel_x = -0.3 = backward (in -X)
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_z = 0.0

        # ── State machine ─────────────────────────────────────────────────
        self.phase = "BACK"
        self.step = 0

        # ── LiDAR ────────────────────────────────────────────────────────────
        self.lidar_box = None

        # ── Diagnostic ─────────────────────────────────────────────────────
        self._last_phase = None
        self._printed_obs = False

    # ══════════════════════════════════════════════════════════════════════════
    # Pose estimation (dead reckoning)
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pose(self, proprio: torch.Tensor) -> None:
        """Integrate robot position from base_lin_vel."""
        base_lin = proprio[0, 0:3].cpu().numpy()
        base_ang = proprio[0, 3:6].cpu().numpy()
        vx, vy = base_lin[0], base_lin[1]
        yaw_rate = base_ang[2]

        cos_y = math.cos(self.est_yaw)
        sin_y = math.sin(self.est_yaw)

        # World frame integration
        self.est_x += (cos_y * vx - sin_y * vy) * self._dt
        self.est_y += (sin_y * vx + cos_y * vy) * self._dt
        self.est_yaw += yaw_rate * self._dt

        while self.est_yaw > math.pi:  self.est_yaw -= 2 * math.pi
        while self.est_yaw < -math.pi: self.est_yaw += 2 * math.pi

    # ══════════════════════════════════════════════════════════════════════════
    # LiDAR
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
    # State machine (coordinate-based using world position)
    # ══════════════════════════════════════════════════════════════════════════

    def _transition(self) -> None:
        p = self.phase
        rx, ry = self.est_x, self.est_y

        if p == "BACK":
            # Back up until robot X < BACK_X
            if rx <= self.BACK_X:
                self.phase = "LEFT"
                self.step = 0

        elif p == "LEFT":
            # Walk to Y > BOX_Y (left side of box)
            if ry >= self.BOX_Y:
                self.phase = "PUSH_RIGHT"
                self.step = 0

        elif p == "PUSH_RIGHT":
            # Push box +X until robot passes box X
            if rx >= -2.8:
                self.phase = "BACK_SIDE"
                self.step = 0

        elif p == "BACK_SIDE":
            # Walk to Y < BOX_Y (right side / behind box)
            if ry <= self.BOX_Y - 0.3:
                self.phase = "PUSH_PIT"
                self.step = 0

        elif p == "PUSH_PIT":
            # Push box into pit until robot reaches PIT_X
            if rx >= self.PIT_X:
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

        # ── Pose update + sensing ───────────────────────────────────────────
        self._update_pose(proprio)
        lb = self._detect_box_lidar(obs)
        self.lidar_box = lb
        self._transition()

        p = self.phase

        # ── Velocity command per phase ────────────────────────────────────
        # Convention (from user testing):
        #   vel_x = forward speed (+X world), vel_y = strafe (+=left, -=right)
        if p == "BACK":
            self._vel_x = -0.3  # backward in -X
            self._vel_y = 0.0
        elif p == "LEFT":
            self._vel_x = 0.0
            self._vel_y = 0.5   # strafe left (+Y)
        elif p == "PUSH_RIGHT":
            self._vel_x = 0.5   # forward (+X)
            self._vel_y = 0.0
        elif p == "BACK_SIDE":
            self._vel_x = 0.0
            self._vel_y = -0.5  # strafe right (-Y)
        elif p == "PUSH_PIT":
            self._vel_x = 0.5
            self._vel_y = 0.0
        elif p == "CROSS":
            self._vel_x = 0.4
            self._vel_y = 0.0
        self._vel_z = 0.0

        action = self._run_policy(obs, action_dim)

        # ── Log ──────────────────────────────────────────────────────────
        if p != self._last_phase:
            lb_str = (f"rng={lb['range']:.2f}" if lb else "none")
            hdg_deg = f"{math.degrees(self._vel_y):+.0f}°"
            print(
                f"[D] phase={p:<12}  robot=({self.est_x:+.2f},{self.est_y:+.2f})  "
                f"lidar=[{lb_str}]  cmd=(fwd={self._vel_x:+.1f}, hdg={hdg_deg})"
            )
            self._last_phase = p

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}