"""
Task D: Push box into pit.

Strategy:
  Phase 1: GO_TO_BOX    → Walk LEFT in Y toward box's Y position
  Phase 2: PUSH_FORWARD → Walk FORWARD to push box toward pit
  Phase 3: CROSS        → Walk across pit

Calibration: None. Initial pose from known spawn positions.
Pose updates: Dead reckoning from base_lin_vel + LiDAR yaw correction.
Box tracking: LiDAR bearing + range (no GT).
Transition: Coordinate-based (with step-based fallback limit).
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

        # ── Timing ─────────────────────────────────────────────────────────────
        self._dt = 0.02  # decimation=4, sim.dt=0.005

        # ── Known initial positions (from env_cfg) ────────────────────────────
        # Robot: (-3, 0, yaw=0), Box: (-3, 1.6), Pit center: x=-0.2
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0
        self._yaw_correction = 0.0
        self._wall_bearing_ref = None

        # ── Box pose (LiDAR triangulation) ────────────────────────────────────
        self.lidar_box = None   # {bearing, range, angular_width}
        self.box_x = -3.0
        self.box_y = 1.6
        self.box_conf = 1.0     # start with high confidence from known init

        # ── Rotation signal ────────────────────────────────────────────────────
        self._rotation_signal = 0.0
        self._prev_lidar_bearing = None
        self._bearing_delta_smoothed = 0.0

        # ── Velocity command (updated per phase) ───────────────────────────────
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._vel_z = 0.0

        # ── State machine ───────────────────────────────────────────────────────
        # 1. GO_TO_BOX: walk LEFT (vel_y=+0.5) until robot Y >= box Y (1.6)
        # 2. PUSH_FORWARD: walk FORWARD (vel_x=+0.5) until robot X >= pit
        # 3. CROSS: walk FORWARD (vel_x=+0.4) across pit
        self.phase = "GO_TO_BOX"
        self.step = 0

        # ── Phase targets ──────────────────────────────────────────────────────
        self.BOX_Y = 1.6       # transition when robot reaches box Y
        self.PIT_X = -0.5      # transition when robot reaches pit

        # ── Step limits (fallback) ─────────────────────────────────────────────
        self.STEP_LIMIT = 500

        # ── Diagnostic ────────────────────────────────────────────────────────
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
        """Detect the box cluster in LiDAR scan."""
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

        # Cluster adjacent outlier bins
        clusters = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                clusters.append((start, prev))
                start = prev = idx
        clusters.append((start, prev))

        # Merge wrap-around
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

            range_score = 1.0 / (1.0 + 0.5 * abs(est_range - 2.0))
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
    # Pose estimation
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pose(self, proprio: torch.Tensor) -> None:
        """Dead reckoning from base_lin_vel."""
        base_lin = proprio[0, 0:3].cpu().numpy()
        base_ang = proprio[0, 3:6].cpu().numpy()
        vx, vy = base_lin[0], base_lin[1]
        yaw_rate = base_ang[2]

        corrected_yaw = self.est_yaw + self._yaw_correction
        cos_y = math.cos(corrected_yaw)
        sin_y = math.sin(corrected_yaw)

        self.est_x += (cos_y * vx - sin_y * vy) * self._dt
        self.est_y += (sin_y * vx + cos_y * vy) * self._dt
        self.est_yaw += yaw_rate * self._dt

        while self.est_yaw > math.pi:  self.est_yaw -= 2 * math.pi
        while self.est_yaw < -math.pi: self.est_yaw += 2 * math.pi

    def _update_yaw_correction(self, lb: dict | None) -> None:
        """LiDAR wall-bearing yaw correction (wide clusters = wall)."""
        if lb is None or lb.get("angular_width", 0) < 1.1:
            return

        world_bearing = self.est_yaw + lb["bearing"]
        if self._wall_bearing_ref is None:
            self._wall_bearing_ref = world_bearing
            self._yaw_correction = 0.0
            return

        expected = self._wall_bearing_ref - self.est_yaw
        drift = lb["bearing"] - expected
        while drift > math.pi:  drift -= 2 * math.pi
        while drift < -math.pi: drift += 2 * math.pi
        self._yaw_correction += 0.04 * drift
        self._yaw_correction = max(-0.5, min(0.5, self._yaw_correction))

    def _update_box_est(self, lb: dict | None) -> None:
        """LiDAR triangulation of box position."""
        if lb is None:
            self.box_conf = max(0.0, self.box_conf * 0.97)
            return

        bearing = lb["bearing"]
        range_m = lb["range"]
        corrected_yaw = self.est_yaw + self._yaw_correction
        world_bearing = corrected_yaw + bearing

        cx = self.est_x + math.cos(world_bearing) * range_m
        cy = self.est_y + math.sin(world_bearing) * range_m

        alpha = 0.15
        if self.box_conf < 0.1:
            self.box_x, self.box_y = cx, cy
        else:
            self.box_x = (1 - alpha) * self.box_x + alpha * cx
            self.box_y = (1 - alpha) * self.box_y + alpha * cy
        self.box_conf = min(1.0, self.box_conf + 0.1)

    def _update_rotation_signal(self, lb: dict | None) -> None:
        """Accumulate LiDAR bearing delta to detect box rotation."""
        if lb is None or self._prev_lidar_bearing is None:
            if lb is not None:
                self._prev_lidar_bearing = lb["bearing"]
            return

        d_bearing = lb["bearing"] - self._prev_lidar_bearing
        while d_bearing > math.pi:  d_bearing -= 2 * math.pi
        while d_bearing < -math.pi: d_bearing += 2 * math.pi

        self._bearing_delta_smoothed = 0.6 * self._bearing_delta_smoothed + 0.4 * d_bearing

        close_factor = 2.0 / max(0.8, lb["range"])
        self._rotation_signal += self._bearing_delta_smoothed * close_factor

        self._prev_lidar_bearing = lb["bearing"]

    # ══════════════════════════════════════════════════════════════════════════
    # Policy interface
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
        """45-dim policy observation (mirrors solution_rl.py)."""
        proprio = obs["proprio"].to(self.device)

        idx = 0
        _ = proprio[:, idx:idx + 3]; idx += 3  # base_lin_vel (unused)
        base_ang_vel  = proprio[:, idx:idx + 3]; idx += 3
        _ = proprio[:, idx:idx + 3]; idx += 3  # velocity_commands_env (overwritten)
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3
        joint_pos_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel_all = proprio[:, idx:idx + action_dim]; idx += action_dim
        actions_all   = proprio[:, idx:idx + action_dim]

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
    # State machine
    # ══════════════════════════════════════════════════════════════════════════

    def _transition(self) -> None:
        p = self.phase
        rx, ry = self.est_x, self.est_y

        if p == "GO_TO_BOX":
            # Walk LEFT (positive Y) until reaching box's Y
            # Only transition when Y is reached (no step fallback — wait for real position)
            if ry >= self.BOX_Y:
                self.phase = "PUSH_FORWARD"
                self.step = 0

        elif p == "PUSH_FORWARD":
            # Walk FORWARD (+X) to push box into pit
            if rx >= self.PIT_X:
                self.phase = "CROSS"
                self.step = 0

        elif p == "CROSS":
            pass

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

        # ── Sensing ───────────────────────────────────────────────────────────
        lb = self._detect_box_lidar(obs)
        self._update_pose(proprio)
        self._update_yaw_correction(lb)
        self._update_box_est(lb)
        self._update_rotation_signal(lb)
        self._transition()

        p = self.phase

        # ── Velocity commands per phase ─────────────────────────────────────
        # Robot frame: +X = forward, +Y = left
        if p == "GO_TO_BOX":
            # Move LEFT (toward box's Y) — NO forward (wall at x=-3)
            self._vel_x, self._vel_y, self._vel_z = 0.0, 0.5, 0.0
        elif p == "PUSH_FORWARD":
            # Push box toward pit — move FORWARD
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0
        elif p == "CROSS":
            self._vel_x, self._vel_y, self._vel_z = 0.4, 0.0, 0.0

        action = self._run_policy(obs, action_dim)

        # ── Log ─────────────────────────────────────────────────────────────
        if p != self._last_phase:
            lb_str = (f"bear={lb['bearing']:+.2f} rng={lb['range']:.2f} aw={lb['angular_width']:.2f}"
                     if lb else "none")
            print(
                f"[D] phase={p:<14} step={self.step:>3}  "
                f"robot=({self.est_x:+.2f},{self.est_y:+.2f},yaw={self.est_yaw:+.2f})  "
                f"box=({self.box_x:+.2f},{self.box_y:+.2f}) conf={self.box_conf:.2f}  "
                f"lidar=[{lb_str}]  rot={self._rotation_signal:+.3f}  "
                f"cmd=({self._vel_x:+.1f},{self._vel_y:+.1f})"
            )
            self._last_phase = p

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}