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
        self.BACK_X = -4.0    # back up farther
        self.PIT_X = 1.5      # cross pit until this X

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

        # ── Step limits per phase (fallback) ───────────────────────────────
        self.BACK_STEPS = 800
        self.LEFT_STEPS = 600
        self.PUSH_RIGHT_STEPS = 1000  # push farther to get box near pit
        self.BACK_SIDE_STEPS = 600
        self.PUSH_PIT_STEPS = 700
        self.CROSS_STEPS = 500

        # ── LiDAR ────────────────────────────────────────────────────────────
        self.lidar_box = None

        # ── Diagnostic ─────────────────────────────────────────────────────
        self._last_phase = None
        self._printed_obs = False

    # ══════════════════════════════════════════════════════════════════════════
    # Pose estimation (dead reckoning)
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pose(self, proprio: torch.Tensor) -> None:
        """Integrate robot position from base_lin_vel (BODY frame convention).

        base_lin_vel is in body frame: vx_body=forward/backward, vy_body=left/right.
        Rotate to world frame using current yaw to accumulate world position.
        This matches the velocity command convention where vel_y=+left.
        """
        base_lin = proprio[0, 0:3].cpu().numpy()
        base_ang = proprio[0, 3:6].cpu().numpy()
        vx_body, vy_body = base_lin[0], base_lin[1]
        yaw_rate = base_ang[2]

        cos_y = math.cos(self.est_yaw)
        sin_y = math.sin(self.est_yaw)

        # Body-to-world rotation (yaw only, robot stays upright)
        world_vx = cos_y * vx_body - sin_y * vy_body
        world_vy = sin_y * vx_body + cos_y * vy_body

        self.est_x += world_vx * self._dt
        self.est_y += world_vy * self._dt
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
        s = self.step
        rx, ry = self.est_x, self.est_y

        if p == "BACK":
            if rx <= self.BACK_X or s >= self.BACK_STEPS:
                self.phase = "LEFT"
                self.step = 0

        elif p == "LEFT":
            if ry >= self.BOX_Y or s >= self.LEFT_STEPS:
                self.phase = "PUSH_RIGHT"
                self.step = 0

        elif p == "PUSH_RIGHT":
            # Push box in +X direction until it reaches the pit area (x >= -0.5)
            # This requires pushing far enough so the box is properly in the pit
            if rx >= -0.5 or s >= self.PUSH_RIGHT_STEPS:
                self.phase = "BACK_SIDE"
                self.step = 0

        elif p == "BACK_SIDE":
            # Move to y < 1.0 (behind the box)
            if ry <= 1.0 or s >= self.BACK_SIDE_STEPS:
                self.phase = "PUSH_PIT"
                self.step = 0

        elif p == "PUSH_PIT":
            if rx >= self.PIT_X or s >= self.PUSH_PIT_STEPS:
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

        if current_score >= 35:
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
        # vel_x = forward speed in +X world, vel_y = strafe (+=left, -=right)
        # heading_command=True → command[2] = TARGET HEADING in world frame
        # Robot will rotate to achieve target heading while moving
        if p == "BACK":
            self._vel_x = -1.0  # fast backward
            self._vel_y = 0.0
            self._vel_z = 0.0   # no heading target during back
        elif p == "LEFT":
            self._vel_x = 0.0
            self._vel_y = 1.0   # fast strafe left
            self._vel_z = 0.0   # maintain current heading
        elif p == "PUSH_RIGHT":
            self._vel_x = 0.8   # moderate forward
            self._vel_y = 0.0
            # Force heading toward +X (0.0) to push in world +X direction
            self._vel_z = 0.0
        elif p == "BACK_SIDE":
            self._vel_x = 0.0
            self._vel_y = -1.0  # fast strafe right
            self._vel_z = 0.0
        elif p == "PUSH_PIT":
            self._vel_x = 0.8
            self._vel_y = 0.0
            self._vel_z = 0.0   # keep facing +X
        elif p == "CROSS":
            self._vel_x = 0.8
            self._vel_y = 0.0
            self._vel_z = 0.0

        action = self._run_policy(obs, action_dim)

        # ── Log ──────────────────────────────────────────────────────────
        if p != self._last_phase:
            lb_str = (f"rng={lb['range']:.2f}" if lb else "none")
            print(
                f"[D] phase={p:<12}  robot=({self.est_x:+.2f},{self.est_y:+.2f},{math.degrees(self.est_yaw):+.0f}°)  "
                f"lidar=[{lb_str}]  cmd=(fwd={self._vel_x:+.1f}, str={self._vel_y:+.1f}, hdg={self._vel_z:+.2f})"
            )
            self._last_phase = p

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}