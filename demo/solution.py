"""
Task D solution: Box rotation for pit crossing.

Strategy:
  Phase 1: TO_BOX      → Walk toward box's Y position
  Phase 2: APPROACH    → Walk to the +Y side of the box (side where we push for rotation)
  Phase 3: ROTATE      → Push against box corner → box rotates ~90°
  Phase 4: ALIGN       → Get behind the rotated box
  Phase 5: PUSH_TO_PIT → Push rotated box until corner bridges pit
  Phase 6: CROSS       → Walk across the pit using the box corner as bridge

Detection:
  - Box detected via LiDAR cluster analysis (outlier detection + angular clustering)
  - Box position triangulated from LiDAR bearing + range + robot pose
  - Rotation detected via accumulated LiDAR bearing delta (box face changes angle)
  - Yaw corrected via LiDAR wall-bearing tracking

No ground-truth access — pure sensor-based estimation.
"""

import os
import math
import torch


class AlgSolution:

    ACTION_SCALE = 0.5
    EE_BODY_NAME_CANDIDATES = ("gripper_base", "piper_gripper_base")
    ARM_JOINT_NAME_CANDIDATES = (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        ["arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5", "arm_joint6"],
    )

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

        # ── Velocity command (per phase) ──────────────────────────────────────
        self._vel_x = 0.5
        self._vel_y = 0.0
        self._vel_z = 0.0

        # ── Timing ────────────────────────────────────────────────────────────
        self._dt = 0.02  # per step (from decimation=4, sim.dt=0.005)

        # ── Robot pose (dead reckoning from velocity commands) ─────────────────
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0
        # Yaw correction from LiDAR wall-bearing
        self._yaw_correction = 0.0
        self._wall_bearing_ref = None   # world bearing of a known wall
        self._yaw_conf_ticks = 0
        self._yaw_drift_ticks = 0

        # ── Box pose (LiDAR triangulation only) ───────────────────────────────
        self.lidar_box = None   # {bearing, range, angular_width}
        self.box_x = -3.0       # world position
        self.box_y = 1.6
        self.box_conf = 0.0     # 0=unknown, 1=confirmed

        # ── Rotation signal (accumulates during ROTATE phase only) ─────────────
        self._rotation_signal = 0.0
        self._prev_lidar_bearing = None
        self._bearing_delta_smoothed = 0.0

        # ── State machine ───────────────────────────────────────────────────────
        self.phase = "TO_BOX"
        self.step = 0
        self._prev_x = -3.0
        self._prev_y = 0.0

        # ── Phase thresholds ───────────────────────────────────────────────────
        # Box starts at (-3, 1.6), pit at x≈-0.8 to 0.2
        # Robot needs to:
        #   1. Get to box's Y (y≈1.6)
        #   2. Move to +Y side (y≈2.4+) for rotation push
        #   3. Push corner → box rotates
        #   4. Push rotated box to pit
        self.BOX_Y = 1.6
        self.SIDE_Y = 2.5      # +Y side (rotation push position)
        self.PIT_CENTER_X = -0.2  # center of pit

        # Rotation detection
        self.ROT_SIG_TARGET = 0.6   # lower threshold — we want actual rotation
        self.ROT_PUSH_STEPS = 100   # push for 2s
        self.ROT_RELEASE_STEPS = 60 # release for 1.2s
        self.MAX_ROT_CYCLES = 12    # max push+release cycles

        # ── Rotation sub-state ────────────────────────────────────────────────
        self._rot_cycles = 0
        self._rot_sub = "push"   # "push" or "release"

        # ── Diagnostic ────────────────────────────────────────────────────────
        self._last_phase = None
        self._printed_obs = False

    # ══════════════════════════════════════════════════════════════════════════
    # LiDAR processing
    # ══════════════════════════════════════════════════════════════════════════

    def _get_lidar_scan(self, obs) -> torch.Tensor | None:
        """Extract horizontal LiDAR scan from obs['extero']."""
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
        """Detect the box from LiDAR scan.

        Process:
          1. Collapse 16 vertical channels → horizontal profile (360 bins).
          2. Find outlier bins (objects vs floor) via percentile threshold.
          3. Cluster adjacent outlier bins.
          4. Score each cluster; return best one.
        """
        scan = self._get_lidar_scan(obs)
        if scan is None or scan.numel() < 32:
            return None

        if not hasattr(self, "_lidar_printed") or not self._lidar_printed:
            self._lidar_printed = True

        flat = scan.flatten()
        finite_mask = flat.isfinite()
        values = flat[finite_mask]

        if values.numel() < 16:
            return None

        # Collapse channels into horizontal profile
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

        # Find outliers (objects vs floor/ceiling)
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

        # Cluster adjacent bins
        clusters = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                clusters.append((start, prev))
                start = prev = idx
        clusters.append((start, prev))

        # Merge wrap-around cluster
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

            # Filter: must be medium width (box-sized, not wall-sized)
            if width < 4 or angular_w < 0.08 or angular_w > 1.2:
                continue

            # Range estimate from angular width (box ~1m wide in Y)
            est_range = 1.0 / max(angular_w, 0.01)
            est_range = max(0.4, min(6.0, est_range))

            # Weighted centroid bearing
            idxs = torch.arange(s, e + 1, device=self.device) % n_bins
            angles = (idxs.float() / float(n_bins - 1)) * (2 * math.pi) - math.pi
            weights = deviation[idxs].clamp_min(1e-4)
            sin_mean = (weights * torch.sin(angles)).sum() / weights.sum()
            cos_mean = (weights * torch.cos(angles)).sum() / weights.sum()
            bearing = math.atan2(sin_mean.item(), cos_mean.item())

            # Score: prefer medium range, forward-ish bearing, moderate width
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
    # Pose estimation (dead reckoning from velocity commands + LiDAR yaw corr)
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pose(self, proprio: torch.Tensor) -> None:
        """Update robot pose via dead reckoning + LiDAR yaw correction."""
        # Get actual robot motion from base_lin_vel in proprio
        base_lin = proprio[0, 0:3].cpu().numpy()
        base_ang = proprio[0, 3:6].cpu().numpy()
        vx, vy = base_lin[0], base_lin[1]
        yaw_rate = base_ang[2]

        # World frame integration with yaw correction
        corrected_yaw = self.est_yaw + self._yaw_correction
        cos_y = math.cos(corrected_yaw)
        sin_y = math.sin(corrected_yaw)

        self.est_x += (cos_y * vx - sin_y * vy) * self._dt
        self.est_y += (sin_y * vx + cos_y * vy) * self._dt
        self.est_yaw += yaw_rate * self._dt

        # Normalize yaw
        while self.est_yaw > math.pi:  self.est_yaw -= 2 * math.pi
        while self.est_yaw < -math.pi: self.est_yaw += 2 * math.pi

    def _update_yaw_correction(self, lb: dict | None) -> None:
        """Correct yaw drift using LiDAR wall-bearing."""
        if lb is None:
            self._yaw_drift_ticks += 1
            self._yaw_conf_ticks = 0
            # Slowly decay correction when no wall detected
            if self._yaw_drift_ticks > 200 and self._wall_bearing_ref is not None:
                self._yaw_correction *= 0.998
            return

        world_bearing = self.est_yaw + lb["bearing"]
        width = lb.get("angular_width", 0.3)

        # Wide cluster = wall (use for yaw correction)
        # Narrow cluster = box (ignore for yaw correction)
        if width > 1.1:
            self._yaw_drift_ticks = 0
            self._yaw_conf_ticks += 1

            if self._wall_bearing_ref is None:
                # First wall contact — establish reference
                self._wall_bearing_ref = world_bearing
                self._yaw_correction = 0.0
            else:
                # Compute drift between expected and actual wall bearing
                expected = self._wall_bearing_ref - self.est_yaw
                drift = lb["bearing"] - expected
                while drift > math.pi:  drift -= 2 * math.pi
                while drift < -math.pi: drift += 2 * math.pi
                self._yaw_correction += 0.04 * drift
                self._yaw_correction = max(-0.5, min(0.5, self._yaw_correction))
        else:
            self._yaw_drift_ticks += 1
            self._yaw_conf_ticks = 0

    def _update_box_est(self, lb: dict | None) -> None:
        """Update box position via LiDAR triangulation."""
        if lb is None:
            # Box not in view — slowly decay confidence
            self.box_conf = max(0.0, self.box_conf * 0.97)
            return

        bearing = lb["bearing"]
        range_m = lb["range"]
        corrected_yaw = self.est_yaw + self._yaw_correction
        world_bearing = corrected_yaw + bearing

        cx = self.est_x + math.cos(world_bearing) * range_m
        cy = self.est_y + math.sin(world_bearing) * range_m

        alpha = 0.2  # trust new measurement
        if self.box_conf < 0.1:
            self.box_x, self.box_y = cx, cy
        else:
            self.box_x = (1 - alpha) * self.box_x + alpha * cx
            self.box_y = (1 - alpha) * self.box_y + alpha * cy
        self.box_conf = min(1.0, self.box_conf + 0.1)

    def _update_rotation_signal(self, lb: dict | None) -> None:
        """Accumulate LiDAR bearing delta to detect box rotation.

        Only meaningful during ROTATE phase when actively pushing.
        Each push shifts the box's face angle → bearing delta accumulates.
        """
        if lb is None or self._prev_lidar_bearing is None:
            if lb is not None:
                self._prev_lidar_bearing = lb["bearing"]
            return

        bearing = lb["bearing"]
        range_m = lb["range"]

        d_bearing = bearing - self._prev_lidar_bearing
        while d_bearing > math.pi:  d_bearing -= 2 * math.pi
        while d_bearing < -math.pi: d_bearing += 2 * math.pi

        # EMA filter
        self._bearing_delta_smoothed = 0.6 * self._bearing_delta_smoothed + 0.4 * d_bearing

        # Only accumulate rotation signal during push phases (close range = stronger)
        close_factor = 2.0 / max(0.8, range_m)  # stronger when close
        self._rotation_signal += self._bearing_delta_smoothed * close_factor

        self._prev_lidar_bearing = bearing

    # ══════════════════════════════════════════════════════════════════════════
    # Policy interface (matches solution_rl.py exactly)
    # ══════════════════════════════════════════════════════════════════════════

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Return velocity command as 2D tensor (num_envs, 3)."""
        num_envs = int(proprio.shape[0])
        cmd = torch.tensor(
            [self._vel_x, self._vel_y, self._vel_z],
            device=self.device, dtype=torch.float32,
        ).view(1, 3)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    def _extract_policy_obs(self, obs, action_dim: int) -> torch.Tensor:
        """Build the 45-dim policy observation. Mirrors solution_rl.py."""
        proprio = obs["proprio"].to(self.device)

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]; idx += 3
        base_ang_vel  = proprio[:, idx:idx + 3]; idx += 3
        _velocity_commands_env = proprio[:, idx:idx + 3]; idx += 3
        projected_gravity = proprio[:, idx:idx + 3]; idx += 3
        joint_pos_all    = proprio[:, idx:idx + action_dim]; idx += action_dim
        joint_vel_all    = proprio[:, idx:idx + action_dim]; idx += action_dim
        actions_all      = proprio[:, idx:idx + action_dim]

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
        """Run the RL policy with current velocity command."""
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
        bx, by = self.box_x, self.box_y

        if p == "TO_BOX":
            # Walk toward box's Y position (forward + left)
            # Transition when robot reaches box's Y band
            if ry >= self.BOX_Y - 0.2:
                self.phase = "APPROACH"
                self.step = 0
                self._prev_y = ry

        elif p == "APPROACH":
            # Move to +Y side of box for rotation push
            # Transition when robot reaches +Y side
            if ry >= self.SIDE_Y:
                self.phase = "ROTATE"
                self.step = 0
                self._rot_cycles = 0
                self._rot_sub = "push"
                self._rotation_signal = 0.0
                self._prev_lidar_bearing = None
                self._bearing_delta_smoothed = 0.0

        elif p == "ROTATE":
            if self._rot_sub == "push":
                # Keep pushing for rotation
                if self._rotation_signal >= self.ROT_SIG_TARGET:
                    self.phase = "ALIGN"
                    self.step = 0
                    self._prev_x = rx
                # Also transition if pushed too far (over-rotated)
                elif rx >= -1.5:
                    self.phase = "ALIGN"
                    self.step = 0
                    self._prev_x = rx
            elif self._rot_sub == "release":
                if self._rotation_signal >= self.ROT_SIG_TARGET:
                    self.phase = "ALIGN"
                    self.step = 0
                    self._prev_x = rx
                else:
                    self._rot_cycles += 1
                    if self._rot_cycles >= self.MAX_ROT_CYCLES:
                        self.phase = "ALIGN"
                        self.step = 0
                        self._prev_x = rx
                    else:
                        self._rot_sub = "push"
                        self.step = 0

        elif p == "ALIGN":
            # Move to x position near box for final push
            if abs(rx - bx) < 0.8:
                self.phase = "PUSH_TO_PIT"
                self.step = 0
                self._prev_x = rx

        elif p == "PUSH_TO_PIT":
            # Push the rotated box toward the pit
            # Stop when robot is near pit center
            if rx >= self.PIT_CENTER_X - 0.3:
                self.phase = "CROSS"
                self.step = 0

        elif p == "CROSS":
            pass  # just keep walking

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

        # ── Update sensing ────────────────────────────────────────────────────
        lb = self._detect_box_lidar(obs)

        self._update_pose(proprio)
        self._update_yaw_correction(lb)
        self._update_box_est(lb)
        self._update_rotation_signal(lb)
        self._transition()

        p = self.phase

        # ── Velocity commands per phase ─────────────────────────────────────
        if p == "TO_BOX":
            # Walk forward + left toward box's Y
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.4, 0.0
        elif p == "APPROACH":
            # Walk forward + left toward +Y side
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.5, 0.0
        elif p == "ROTATE":
            if self._rot_sub == "push":
                # Forward + slightly right (push against box's corner)
                self._vel_x, self._vel_y, self._vel_z = 0.5, -0.2, 0.0
            else:
                # Release: stop forward, slight retreat
                self._vel_x, self._vel_y, self._vel_z = 0.1, 0.0, 0.0
        elif p == "ALIGN":
            # Move toward box for final push
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0
        elif p == "PUSH_TO_PIT":
            # Push box toward pit
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0
        elif p == "CROSS":
            # Walk across pit
            self._vel_x, self._vel_y, self._vel_z = 0.4, 0.0, 0.0

        action = self._run_policy(obs, action_dim)

        # ── Log on phase change ────────────────────────────────────────────
        if p != self._last_phase:
            lb_str = (f"bear={lb['bearing']:+.2f} rng={lb['range']:.2f} aw={lb['angular_width']:.2f}"
                     if lb else "none")
            print(
                f"[D] phase={p:<14} step={self.step:>3}  "
                f"robot=({self.est_x:+.2f},{self.est_y:+.2f},yaw={self.est_yaw:+.2f})  "
                f"box=({self.box_x:+.2f},{self.box_y:+.2f}) conf={self.box_conf:.2f}  "
                f"lidar=[{lb_str}]  rot_sig={self._rotation_signal:+.3f}  "
                f"yaw_corr={self._yaw_correction:+.3f}  "
                f"cmd=({self._vel_x:+.1f},{self._vel_y:+.1f})"
            )
            self._last_phase = p

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}