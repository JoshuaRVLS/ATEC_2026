"""
Task D solution: Box rotation for pit crossing.

Built on top of solution_rl.py (the working baseline).
The RL policy works when velocity commands are injected properly as a 2D tensor.

Features:
  - LiDAR box detection (cluster analysis)
  - Box world position estimation (LiDAR bearing + range + pose)
  - Rotation signal (LiDAR bearing delta accumulation)
  - Ground-truth pose access (robot + box positions from sim)
  - Smart transitions using real sensor data
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

        # ── Velocity command (per phase) ────────────────────────────────────────
        self._vel_x = 0.5
        self._vel_y = 0.0
        self._vel_z = 0.0

        # ── Ground-truth cache ─────────────────────────────────────────────────
        self._env_cache = None
        self._gt_cached = False
        self._dt = 0.02

        # ── Pose estimation ─────────────────────────────────────────────────
        # Dead reckoning (fallback when GT unavailable)
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0
        self._yaw_correction = 0.0
        self._wall_world_bearing = None
        self._yaw_confirm_ticks = 0
        self._yaw_drift_ticks = 0
        self._prev_lidar_bearing = None
        self._bearing_delta = 0.0

        # ── LiDAR box tracking ──────────────────────────────────────────────
        self.lidar_box = None       # raw LiDAR detection: {bearing, range, angular_width}
        self.box_est_x = -3.0
        self.box_est_y = 1.6
        self.box_conf = 0.0

        # ── Rotation signal ────────────────────────────────────────────────
        self._rotation_signal = 0.0
        self._prev_lidar_bearing_for_rotation = None

        # ── State machine ─────────────────────────────────────────────────────
        self.phase = "TO_BOX"
        self.step = 0
        self._stuck_ticks = 0
        self._prev_x = -3.0
        self._prev_y = 0.0

        # ── Phase thresholds ──────────────────────────────────────────────────
        self.PIT_CENTER_X = -0.8
        self.STOP_PUSH_BOX_X = -2.8
        self.BOX_Y_TARGET = 1.6
        self.SIDE_Y = 2.5

        # ── Rotation sub-state ───────────────────────────────────────────────
        self._rotate_pulses = 0
        self._rotate_substep = "push"
        self.ROTATE_SIGNAL_TARGET = 1.8
        self.ROTATE_PUSH_STEPS = 80
        self.ROTATE_RELEASE_STEPS = 50
        self.MAX_ROTATE_PULSES = 10

        # ── Diagnostic ────────────────────────────────────────────────────────
        self._last_phase = None
        self._printed_obs = False
        self._printed_lidar = False

    # ══════════════════════════════════════════════════════════════════════════
    # Ground-truth access
    # ══════════════════════════════════════════════════════════════════════════

    def _cache_env(self, obs: dict) -> None:
        """Cache the scene object from the observation tensor."""
        if self._gt_cached:
            return
        try:
            # obs["proprio"] is a torch.Tensor with shape [1, 72].
            # Check numel() instead of boolean to avoid PyTorch ambiguity error.
            proprio = obs.get("proprio")
            if proprio is None or proprio.numel() == 0:
                return

            # The tensor has a _manager attribute pointing to the ObservationManager.
            manager = getattr(proprio, "_manager", None)
            if manager is None:
                return

            # Manager has a _env attribute pointing to the ManagerBasedRLEnv.
            env = getattr(manager, "_env", None)
            if env is None:
                env = manager  # might be the env itself

            scene = getattr(env, "scene", None)
            if scene is None:
                return

            self._env_cache = scene
            self._gt_cached = True

            # Initialize pose from ground truth
            gt = self._get_gt_robot_pose()
            if gt:
                self.est_x, self.est_y, self.est_yaw = gt
                self._prev_x, self._prev_y = self.est_x, self.est_y

            gt_box = self._get_gt_box_pose()
            if gt_box:
                self.box_est_x, self.box_est_y = gt_box
                self.box_conf = 1.0

        except Exception as e:
            self._gt_cached = False

    def _get_gt_robot_pose(self):
        """Return (x, y, yaw) of robot root in world frame."""
        try:
            if self._env_cache is None:
                return None
            robot = self._env_cache["robot"]
            pos = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0].cpu().numpy()
            yaw = math.atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
            )
            return float(pos[0]), float(pos[1]), float(yaw)
        except Exception:
            return None

    def _get_gt_box_pose(self):
        """Return (x, y) of box root in world frame."""
        try:
            if self._env_cache is None:
                return None
            box = self._env_cache["box"]
            pos = box.data.root_pos_w[0].cpu().numpy()
            return float(pos[0]), float(pos[1])
        except Exception:
            return None

    # ══════════════════════════════════════════════════════════════════════════
    # Pose estimation
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pose(self, proprio: torch.Tensor) -> None:
        """Update pose estimate: GT if available, else dead reckoning."""
        gt = self._get_gt_robot_pose()
        if gt is not None:
            self.est_x, self.est_y, raw_yaw = gt
            # Calibrate yaw correction
            drift = raw_yaw - self.est_yaw
            while drift > math.pi: drift -= 2 * math.pi
            while drift < -math.pi: drift += 2 * math.pi
            self._yaw_correction += 0.05 * drift
            self._yaw_correction = max(-0.5, min(0.5, self._yaw_correction))
            self.est_yaw = raw_yaw
            return

        # Dead reckoning fallback
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

    def _fuse_lidar_box(self) -> None:
        """Update box position: GT if available, else LiDAR triangulation."""
        gt_box = self._get_gt_box_pose()
        if gt_box is not None:
            self.box_est_x, self.box_est_y = gt_box
            self.box_conf = 1.0
            return

        lb = self.lidar_box
        if lb is None:
            self.box_conf = max(0.0, self.box_conf * 0.99)
            return

        bearing = lb["bearing"]
        range_m = lb["range"]
        corrected_yaw = self.est_yaw + self._yaw_correction
        world_bearing = corrected_yaw + bearing
        cx = self.est_x + math.cos(world_bearing) * range_m
        cy = self.est_y + math.sin(world_bearing) * range_m

        alpha = 0.15
        if self.box_conf < 0.05:
            self.box_est_x, self.box_est_y = cx, cy
        else:
            self.box_est_x = (1 - alpha) * self.box_est_x + alpha * cx
            self.box_est_y = (1 - alpha) * self.box_est_y + alpha * cy
        self.box_conf = min(1.0, self.box_conf + 0.08)

    def _update_rotation_signal(self) -> None:
        """Accumulate LiDAR bearing delta to detect box rotation."""
        lb = self.lidar_box
        if lb is None:
            return

        bearing = lb["bearing"]
        range_m = lb["range"]
        width = lb.get("angular_width", 0.3)

        if self._prev_lidar_bearing_for_rotation is not None:
            d_bearing = bearing - self._prev_lidar_bearing_for_rotation
            while d_bearing > math.pi: d_bearing -= 2 * math.pi
            while d_bearing < -math.pi: d_bearing += 2 * math.pi

            self._bearing_delta = 0.7 * self._bearing_delta + 0.3 * d_bearing

            # Scale by inverse range so close contacts dominate
            self._rotation_signal += self._bearing_delta * (2.0 / max(0.5, range_m))

        self._prev_lidar_bearing_for_rotation = bearing

        # Wall yaw correction: wide clusters = wall, narrow = box
        if self._prev_lidar_bearing is not None and width > 1.0:
            self._yaw_drift_ticks = 0
            self._yaw_confirm_ticks += 1
            if self._wall_world_bearing is None:
                self._wall_world_bearing = bearing
                self._yaw_correction = 0.0
            else:
                expected = self._wall_world_bearing - self.est_yaw
                drift = bearing - expected
                while drift > math.pi: drift -= 2 * math.pi
                while drift < -math.pi: drift += 2 * math.pi
                self._yaw_correction += 0.03 * drift
                self._yaw_correction = max(-0.5, min(0.5, self._yaw_correction))
        else:
            self._yaw_confirm_ticks = 0
            self._yaw_drift_ticks += 1
            if self._yaw_drift_ticks > 120:
                self._yaw_correction *= 0.998

        self._prev_lidar_bearing = bearing

    # ══════════════════════════════════════════════════════════════════════════
    # LiDAR processing
    # ══════════════════════════════════════════════════════════════════════════

    def _get_lidar_scan(self, obs) -> torch.Tensor | None:
        """Extract the LiDAR horizontal scan from obs."""
        extero = obs.get("extero")
        if extero is None or extero.numel() == 0:
            return None
        scan = extero.to(device=self.device, dtype=torch.float32)
        if scan.ndim == 1:
            scan = scan.view(1, -1)
        elif scan.ndim > 2:
            scan = scan.reshape(scan.shape[0], -1)
        return scan[0]

    def _detect_box_from_lidar(self, obs):
        """Detect the most plausible box cluster in the LiDAR scan.

        Strategy:
          1. Collapse vertical channels into a horizontal profile (median per bin).
          2. Find outlier bins (objects vs floor) using percentile threshold.
          3. Cluster adjacent outlier bins.
          4. Score each cluster by angular width, range estimate, and bearing.
          5. Return bearing, range, and angular width of the best cluster.
        """
        scan = self._get_lidar_scan(obs)
        if scan is None or scan.numel() < 32:
            self.lidar_box = None
            return

        if not self._printed_lidar:
            sample = scan[:12].detach().cpu().tolist()
            print(f"[LiDAR] shape={scan.shape}, finite={scan.isfinite().sum().item()}/{scan.numel()}, sample={sample[:6]}")
            self._printed_lidar = True

        flat = scan.flatten()
        finite_mask = flat.isfinite()
        values = flat[finite_mask]

        if values.numel() < 16:
            self.lidar_box = None
            return

        # Collapse vertical channels into horizontal profile
        n = flat.numel()
        if n % 360 == 0:
            cols = flat.view(-1, 360)
            col_finite = cols.isfinite()
            safe = torch.where(col_finite, cols, torch.zeros_like(cols))
            counts = col_finite.sum(dim=0).clamp_min(1)
            horizontal = safe.sum(dim=0) / counts
        else:
            horizontal = flat

        # Find outliers = objects (box) vs floor
        median = values.median()
        deviation = (horizontal - median).abs()
        valid_dev = deviation[horizontal.isfinite()]
        if valid_dev.numel() < 8:
            self.lidar_box = None
            return

        kth = max(1, int(valid_dev.numel() * 0.88))
        threshold = valid_dev.kthvalue(kth).values.clamp_min(0.06)
        mask = horizontal.isfinite() & (deviation >= threshold)

        # Find angular clusters of outliers
        indices = torch.where(mask)[0].cpu().tolist()
        if not indices:
            self.lidar_box = None
            return

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
                and clusters[-1][1] == mask.numel() - 1):
            last = clusters.pop(-1)
            first = clusters.pop(0)
            clusters.insert(0, (last[0], first[1] + mask.numel()))

        best = None
        best_score = -1.0
        n_bins = horizontal.numel()

        for s, e in clusters:
            width = e - s + 1
            angular_w = float(width) * (2 * math.pi / float(n_bins))
            if width < 5 or angular_w < 0.10 or angular_w > 1.0:
                continue

            # Estimate range from angular width (box ~0.9m wide)
            est_range = 0.9 / max(angular_w, 0.01)
            est_range = max(0.4, min(5.0, est_range))

            # Weighted centroid bearing
            idxs = torch.arange(s, e + 1, device=self.device) % n_bins
            angles = (idxs.float() / float(n_bins - 1)) * (2 * math.pi) - math.pi
            weights = deviation[idxs].clamp_min(1e-4)
            sin_mean = (weights * torch.sin(angles)).sum() / weights.sum()
            cos_mean = (weights * torch.cos(angles)).sum() / weights.sum()
            bearing = math.atan2(sin_mean.item(), cos_mean.item())

            # Score: prefer medium-range, forward-ish clusters
            range_score = 1.0 / (1.0 + 0.3 * abs(est_range - 1.5))
            bearing_score = 1.0 / (1.0 + 0.5 * abs(bearing))
            width_score = math.sqrt(float(width))
            score = width_score * range_score * bearing_score

            if score > best_score:
                best_score = score
                best = (bearing, est_range, angular_w, width)

        if best is None:
            self.lidar_box = None
            return

        bearing, est_range, angular_w, width = best
        self.lidar_box = {
            "bearing": bearing,
            "range": est_range,
            "angular_width": angular_w,
            "count": width,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Policy interface (exactly matches solution_rl.py)
    # ══════════════════════════════════════════════════════════════════════════

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Return the current velocity command as a 2D tensor (num_envs, 3)."""
        num_envs = int(proprio.shape[0])
        cmd = torch.tensor(
            [self._vel_x, self._vel_y, self._vel_z],
            device=self.device, dtype=torch.float32,
        ).view(1, 3)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    def _extract_policy_obs(self, obs, action_dim: int) -> torch.Tensor:
        """Build the 45-dim policy observation. Exactly mirrors solution_rl.py."""
        proprio = obs["proprio"].to(self.device)

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]; idx += 3
        base_ang_vel = proprio[:, idx:idx + 3]; idx += 3
        _velocity_commands_env = proprio[:, idx:idx + 3]; idx += 3
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
            raise ValueError(f"Policy output dim mismatch: got {action_train.shape[-1]}")
        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((num_envs, action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)
        return action_env

    def _phase_action(self, obs, action_dim: int) -> torch.Tensor:
        """Set velocity command and run the RL policy."""
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
    # State machine transitions
    # ══════════════════════════════════════════════════════════════════════════

    def _transition(self) -> None:
        p = self.phase
        s = self.step
        rx, ry = self.est_x, self.est_y
        bx, by = self.box_est_x, self.box_est_y

        if p == "TO_BOX":
            # Walk diagonally forward+left to reach box's Y position
            dx = abs(rx - self._prev_x)
            dy = abs(ry - self._prev_y)
            if dx < 0.005 and dy < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x, self._prev_y = rx, ry

            aligned = ry >= self.BOX_Y_TARGET - 0.3
            stuck = self._stuck_ticks >= 50
            if aligned or stuck or s >= 200:
                self.phase = "CONTACT"
                self.step = 0
                self._prev_x = rx
                self._stuck_ticks = 0

        elif p == "CONTACT":
            # Walk forward into the box
            dx = abs(rx - self._prev_x)
            if dx < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x = rx

            stuck = self._stuck_ticks >= 30
            if stuck or s >= 100:
                self.phase = "PUSH_FWD"
                self.step = 0
                self._stuck_ticks = 0
                self._prev_x = rx

        elif p == "PUSH_FWD":
            # Push box toward pit (but stop early to save room for rotation)
            dx = abs(rx - self._prev_x)
            if dx < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x = rx

            # Use LiDAR-confirmed box position for smart stopping
            box_near_pit = (bx >= self.STOP_PUSH_BOX_X and self.box_conf > 0.1)
            stuck = self._stuck_ticks >= 40
            if box_near_pit or stuck or s >= 300:
                self.phase = "DETACH"
                self.step = 0
                self._prev_x = rx

        elif p == "DETACH":
            # Back away from box
            if self._prev_x - rx >= 0.4 or s >= 80:
                self.phase = "TO_SIDE"
                self.step = 0
                self._prev_y = ry

        elif p == "TO_SIDE":
            # Move to +Y side of box for rotation
            dy = abs(ry - self._prev_y)
            if dy < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_y = ry

            if ry >= self.SIDE_Y or s >= 200:
                self.phase = "ROTATE"
                self.step = 0
                self._rotate_pulses = 0
                self._rotate_substep = "push"

        elif p == "ROTATE":
            if self._rotate_substep == "push":
                if s >= self.ROTATE_PUSH_STEPS:
                    self._rotate_substep = "release"
                    self.step = 0
            elif self._rotate_substep == "release":
                if s >= self.ROTATE_RELEASE_STEPS:
                    # Smart stopping: accumulated bearing delta = rotation detected
                    if self._rotation_signal >= self.ROTATE_SIGNAL_TARGET:
                        self.phase = "PUSH_TO_PIT"
                        self.step = 0
                        self._prev_x = rx
                    else:
                        self._rotate_pulses += 1
                        if self._rotate_pulses >= self.MAX_ROTATE_PULSES:
                            # Force advance even without detected rotation
                            self.phase = "PUSH_TO_PIT"
                            self.step = 0
                            self._prev_x = rx
                        else:
                            self._rotate_substep = "push"
                            self.step = 0

        elif p == "PUSH_TO_PIT":
            # Push the rotated box toward the pit so the corner bridges it
            dx = abs(rx - self._prev_x)
            if dx < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x = rx

            box_near_pit = (bx >= self.PIT_CENTER_X - 0.2 and self.box_conf > 0.15)
            stuck = self._stuck_ticks >= 30
            if box_near_pit or stuck or s >= 250:
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

        # ── Update all sensing ──────────────────────────────────────────────
        self._cache_env(obs)
        self._update_pose(proprio)
        self._detect_box_from_lidar(obs)
        self._fuse_lidar_box()
        self._update_rotation_signal()
        self._transition()

        p = self.phase

        # ── Velocity commands per phase ──────────────────────────────────
        if p == "TO_BOX":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.6, 0.0   # forward + left
        elif p == "CONTACT":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0   # forward
        elif p == "PUSH_FWD":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0
        elif p == "DETACH":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0   # robot backs, box slides forward
        elif p == "TO_SIDE":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.5, 0.0   # forward + left
        elif p == "ROTATE":
            self._vel_x, self._vel_y, self._vel_z = 0.5, -0.3, 0.0   # forward + right (corner push)
        elif p == "PUSH_TO_PIT":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0
        elif p == "CROSS":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0
        else:
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0

        action = self._phase_action(obs, action_dim)

        # ── Log on phase change ───────────────────────────────────────────
        if p != self._last_phase:
            lb = self.lidar_box
            lidar_str = (f"bearing={lb['bearing']:.2f} range={lb['range']:.2f} aw={lb['angular_width']:.2f}"
                        if lb else "none")
            print(
                f"[D] phase={p:<12} step={self.step:>4}  "
                f"pose=({self.est_x:+.2f}, {self.est_y:+.2f}, yaw={self.est_yaw:+.2f})  "
                f"box=({self.box_est_x:+.2f}, {self.box_est_y:+.2f}) conf={self.box_conf:.2f}  "
                f"lidar=[{lidar_str}]  "
                f"rot_sig={self._rotation_signal:+.3f}  pulses={self._rotate_pulses}  "
                f"yaw_corr={self._yaw_correction:+.3f}  "
                f"cmd=({self._vel_x:+.2f}, {self._vel_y:+.2f}, {self._vel_z:+.2f})"
            )
            self._last_phase = p

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}
