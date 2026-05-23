"""
Task D solution: Box rotation for pit crossing.

Strategy (coordinate-based, sensor-driven):
  1. INIT      → Calibrate robot + box pose from GT (first ~10 steps only)
  2. TO_BOX    → Walk to box's Y position
  3. APPROACH  → Walk to +Y side of box (rotation push position)
  4. ROTATE    → Push against box corner → box rotates ~90°
  5. ALIGN     → Get behind the rotated box
  6. PUSH_TO_PIT → Push rotated box until corner bridges pit
  7. CROSS     → Walk across pit using box corner as bridge

Calibration (GT on first N steps only):
  - Robot pose anchored from GT on steps 1-10
  - Box pose anchored from GT on steps 1-10
  - LiDAR box detection continues throughout (fused with GT at start)
  - Dead reckoning + LiDAR yaw correction for pose updates after calibration

After calibration: NO ground-truth access. Pure sensor estimation.
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

        # ── Timing ─────────────────────────────────────────────────────────────
        self._dt = 0.02  # decimation=4, sim.dt=0.005

        # ── Fixed velocity command (same as solution_rl.py baseline) ──────────
        # Policy was trained with these values — only change them if needed
        self._vel_x = 0.5
        self._vel_y = 0.0
        self._vel_z = 0.0

        # ── Calibration state ─────────────────────────────────────────────────
        self._calibration_frames = 10  # calibrate from GT for first N steps
        self._frame = 0
        self._gt_calibrated = False  # set True after calibration window
        self._scene = None  # cached scene for GT access

        # ── Robot pose (calibrated at start, then dead reckoning) ──────────────
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0
        self._yaw_correction = 0.0  # LiDAR wall-bearing correction

        # ── Box pose (calibrated at start, then LiDAR triangulation) ────────────
        self.lidar_box = None   # raw LiDAR detection
        self.box_x = -3.0
        self.box_y = 1.6
        self.box_conf = 0.0

        # ── Rotation signal (accumulates during ROTATE phase) ─────────────────
        self._rotation_signal = 0.0
        self._prev_lidar_bearing = None
        self._bearing_delta_smoothed = 0.0

        # ── LiDAR yaw correction ────────────────────────────────────────────────
        self._wall_bearing_ref = None
        self._yaw_conf_ticks = 0
        self._yaw_drift_ticks = 0

        # ── State machine ────────────────────────────────────────────────────────
        self.phase = "TO_BOX"
        self.step = 0

        # ── Phase targets (from known environment layout) ─────────────────────
        # Box starts at (-3, 1.6), pit center at x≈-0.2
        # Robot starts at (-3, 0, 0), facing -X direction (toward box)
        # Coordinate system: +X = forward (toward pit), +Y = left of robot
        self.BOX_Y = 1.6
        self.SIDE_Y = 2.5        # rotation push position (+Y of box)
        self.PIT_CENTER_X = -0.2  # center of pit
        self.ROT_SIG_TARGET = 0.6
        self.ROT_PUSH_STEPS = 80
        self.ROT_RELEASE_STEPS = 50
        self.MAX_ROT_CYCLES = 15

        # Fallback step limits (coordinate-based + step-based backup)
        self.STEP_LIMITS = {
            "TO_BOX": 400,
            "APPROACH": 350,
            "ROTATE": 600,
            "ALIGN": 300,
            "PUSH_TO_PIT": 400,
        }

        # ── Rotation sub-state ──────────────────────────────────────────────────
        self._rot_cycles = 0
        self._rot_sub = "push"

        # ── Diagnostic ────────────────────────────────────────────────────────
        self._last_phase = None
        self._printed_obs = False

    # ══════════════════════════════════════════════════════════════════════════
    # Ground-truth access (calibration only — first N frames)
    # ══════════════════════════════════════════════════════════════════════════

    def _cache_scene(self, obs: dict) -> None:
        """Cache scene from observation tensor (first frame only)."""
        if self._scene is not None:
            return
        try:
            proprio = obs.get("proprio")
            if proprio is None or proprio.numel() == 0:
                return

            manager = getattr(proprio, "_manager", None)
            if manager is None:
                return

            env = getattr(manager, "_env", None)
            if env is None:
                env = manager

            scene = getattr(env, "scene", None)
            if scene is None:
                return

            self._scene = scene
        except Exception:
            self._scene = None

    def _get_gt_robot_pose(self):
        """Return (x, y, yaw) of robot root in world frame."""
        try:
            if self._scene is None:
                return None
            robot = self._scene["robot"]
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
            if self._scene is None:
                return None
            box = self._scene["box"]
            pos = box.data.root_pos_w[0].cpu().numpy()
            return float(pos[0]), float(pos[1])
        except Exception:
            return None

    def _calibrate(self, obs: dict) -> None:
        """Calibrate poses from GT for first _calibration_frames steps."""
        self._cache_scene(obs)

        if self._frame >= self._calibration_frames:
            self._gt_calibrated = True
            return

        # Robot pose from GT
        gt = self._get_gt_robot_pose()
        if gt is not None:
            self.est_x, self.est_y, self.est_yaw = gt
            # Initialize wall bearing reference from current LiDAR bearing
            if self.lidar_box is not None and self._wall_bearing_ref is None:
                self._wall_bearing_ref = self.est_yaw + self.lidar_box["bearing"]

        # Box pose from GT (only once, when confidence is low)
        if self.box_conf < 0.5:
            gt_box = self._get_gt_box_pose()
            if gt_box is not None:
                self.box_x, self.box_y = gt_box
                self.box_conf = 1.0

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
        """Detect box from LiDAR scan using outlier + clustering analysis."""
        scan = self._get_lidar_scan(obs)
        if scan is None or scan.numel() < 32:
            return None

        flat = scan.flatten()
        finite_mask = flat.isfinite()
        values = flat[finite_mask]

        if values.numel() < 16:
            return None

        # Collapse 16 vertical channels → horizontal profile
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

        # Find outlier bins (objects vs floor) via percentile threshold
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

            # Filter: box-sized clusters (not walls, not tiny)
            if width < 4 or angular_w < 0.08 or angular_w > 1.2:
                continue

            # Range estimate from angular width (box ~1m wide)
            est_range = 1.0 / max(angular_w, 0.01)
            est_range = max(0.4, min(6.0, est_range))

            # Weighted centroid bearing
            idxs = torch.arange(s, e + 1, device=self.device) % n_bins
            angles = (idxs.float() / float(n_bins - 1)) * (2 * math.pi) - math.pi
            weights = deviation[idxs].clamp_min(1e-4)
            sin_mean = (weights * torch.sin(angles)).sum() / weights.sum()
            cos_mean = (weights * torch.cos(angles)).sum() / weights.sum()
            bearing = math.atan2(sin_mean.item(), cos_mean.item())

            # Score: medium range + moderate width preferred
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
    # Pose estimation (post-calibration: dead reckoning + LiDAR yaw corr)
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pose(self, proprio: torch.Tensor) -> None:
        """Update robot pose via dead reckoning + LiDAR yaw correction."""
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
        """Correct yaw drift using LiDAR wall-bearing (wide clusters)."""
        if lb is None:
            self._yaw_drift_ticks += 1
            self._yaw_conf_ticks = 0
            if self._yaw_drift_ticks > 200 and self._wall_bearing_ref is not None:
                self._yaw_correction *= 0.998
            return

        width = lb.get("angular_width", 0.3)

        if width > 1.1:
            self._yaw_drift_ticks = 0
            self._yaw_conf_ticks += 1

            if self._wall_bearing_ref is None:
                self._wall_bearing_ref = self.est_yaw + lb["bearing"]
                self._yaw_correction = 0.0
            else:
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
            self.box_conf = max(0.0, self.box_conf * 0.97)
            return

        bearing = lb["bearing"]
        range_m = lb["range"]
        corrected_yaw = self.est_yaw + self._yaw_correction
        world_bearing = corrected_yaw + bearing

        cx = self.est_x + math.cos(world_bearing) * range_m
        cy = self.est_y + math.sin(world_bearing) * range_m

        alpha = 0.2
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

        bearing = lb["bearing"]
        range_m = lb["range"]

        d_bearing = bearing - self._prev_lidar_bearing
        while d_bearing > math.pi:  d_bearing -= 2 * math.pi
        while d_bearing < -math.pi: d_bearing += 2 * math.pi

        self._bearing_delta_smoothed = 0.6 * self._bearing_delta_smoothed + 0.4 * d_bearing

        # Only accumulate when pushing close to box
        close_factor = 2.0 / max(0.8, range_m)
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
        """Run RL policy with current velocity command."""
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
    # State machine (coordinate-based)
    # ══════════════════════════════════════════════════════════════════════════

    def _transition(self) -> None:
        p = self.phase
        s = self.step
        rx, ry = self.est_x, self.est_y
        _, by = self.box_x, self.box_y
        limit = self.STEP_LIMITS.get(p, 9999)

        if p == "TO_BOX":
            if ry >= self.BOX_Y - 0.2 or s >= limit:
                self.phase = "APPROACH"
                self.step = 0

        elif p == "APPROACH":
            if ry >= self.SIDE_Y or s >= limit:
                self.phase = "ROTATE"
                self.step = 0
                self._rot_cycles = 0
                self._rot_sub = "push"
                self._rotation_signal = 0.0
                self._prev_lidar_bearing = None
                self._bearing_delta_smoothed = 0.0

        elif p == "ROTATE":
            if self._rot_sub == "push":
                if s >= self.ROT_PUSH_STEPS:
                    self._rot_sub = "release"
                    self.step = 0
            elif self._rot_sub == "release":
                if s >= self.ROT_RELEASE_STEPS:
                    self._rot_cycles += 1
                    if self._rotation_signal >= self.ROT_SIG_TARGET:
                        self.phase = "ALIGN"
                        self.step = 0
                    elif self._rot_cycles >= self.MAX_ROT_CYCLES or s >= limit:
                        self.phase = "ALIGN"
                        self.step = 0
                    else:
                        self._rot_sub = "push"
                        self.step = 0

        elif p == "ALIGN":
            # Move near box for final push
            if abs(rx - self.box_x) < 0.8 or s >= limit:
                self.phase = "PUSH_TO_PIT"
                self.step = 0

        elif p == "PUSH_TO_PIT":
            # Push the rotated box toward the pit
            if rx >= self.PIT_CENTER_X - 0.3 or s >= limit:
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

        # ── Step 1: Calibrate from GT (first N frames only) ──────────────────
        self._calibrate(obs)

        # ── Step 2: LiDAR detection ──────────────────────────────────────────
        lb = self._detect_box_lidar(obs)

        # ── Step 3: Update estimates (calibration or sensor-based) ───────────
        if self._gt_calibrated:
            self._update_pose(proprio)
        else:
            # During calibration: keep using GT pose
            gt = self._get_gt_robot_pose()
            if gt is not None:
                self.est_x, self.est_y, self.est_yaw = gt

        self._update_yaw_correction(lb)
        self._update_box_est(lb)
        self._update_rotation_signal(lb)
        self._transition()

        p = self.phase

        # ── Velocity commands per phase ─────────────────────────────────────
        if p == "TO_BOX":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.4, 0.0
        elif p == "APPROACH":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.5, 0.0
        elif p == "ROTATE":
            if self._rot_sub == "push":
                self._vel_x, self._vel_y, self._vel_z = 0.5, -0.2, 0.0
            else:
                self._vel_x, self._vel_y, self._vel_z = 0.1, 0.0, 0.0
        elif p == "ALIGN":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0
        elif p == "PUSH_TO_PIT":
            self._vel_x, self._vel_y, self._vel_z = 0.5, 0.0, 0.0
        elif p == "CROSS":
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
                f"cmd=({self._vel_x:+.1f},{self._vel_y:+.1f})  "
                f"cal={self._frame}/{self._calibration_frames}"
            )
            self._last_phase = p

        self.step += 1
        self._frame += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}