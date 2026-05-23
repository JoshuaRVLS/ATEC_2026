"""
Task D solution: Box rotation for pit crossing.

The box (0.8 x 1.0 x 0.6m) must be rotated ~90° so its corner bridges
the pit, allowing the robot to cross without hitting the invisible barrier.

Key insight: pushing the box straight just translates it. To rotate it, the
robot must push from the side at an offset from the box's center — this
creates a torque. Repeated corner pushes gradually rotate the box until a
corner bridges the pit.
"""

import os
import math
import torch


class AlgSolution:

    ACTION_SCALE = 0.5
    LEG_ACTION_DIM = 12
    ARM_ACTION_DIM = 8

    # ── Fixed velocities for each phase ────────────────────────────────────────
    # (lin_x, lin_y, ang_z) in training command space
    VEL = {
        "backup":        (0.0,  0.0,  0.0),   # overridden below
        "move_to_lane":  (0.0,  0.0,  0.0),   # overridden below
        "approach":      (0.0,  0.0,  0.0),   # overridden below
        "push_forward":  (0.0,  0.0,  0.0),   # overridden below
        "detach":        (0.0,  0.0,  0.0),   # overridden below
        "side_push":     (0.0,  0.0,  0.0),   # overridden below
        "release":       (0.0,  0.0,  0.0),   # overridden below
        "cross":         (0.5,  0.0,  0.0),
    }

    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'policy.pt')
        self.device = 'cuda'

        # Load locomotion baseline (provides a working walking gait)
        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_joint_indices = list(range(self.LEG_ACTION_DIM))
        self.arm_joint_indices = list(range(self.LEG_ACTION_DIM, self.LEG_ACTION_DIM + self.ARM_ACTION_DIM))

        # Training → environment action scale (from rl_utils normalization)
        self.train_to_env = torch.tensor([
            0.25, 0.5, 0.5, 0.25, 0.5, 0.5,
            0.25, 0.5, 0.5, 0.25, 0.5, 0.5,
        ], device=self.device, dtype=torch.float32).view(1, -1)

        self.env_to_train = torch.tensor([
            4.0, 2.0, 2.0, 4.0, 2.0, 2.0,
            4.0, 2.0, 2.0, 4.0, 2.0, 2.0,
        ], device=self.device, dtype=torch.float32).view(1, -1)

        # Default arm action (zero)
        self.arm_default = torch.zeros((1, self.ARM_ACTION_DIM), device=self.device, dtype=torch.float32)

        # Velocity command sent to the locomotion policy
        self._vel_cmd = torch.tensor([0.5, 0.0, 0.0], device=self.device, dtype=torch.float32).view(1, 3)

        # ── State machine ─────────────────────────────────────────────────────
        self.phase = "BACKUP"
        self.step = 0

        # ── Pose estimation (dead reckoning) ──────────────────────────────────
        self.dt = 0.02
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0
        self._prev_x = self.est_x
        self._prev_y = self.est_y
        self._prev_yaw = self.est_yaw

        # ── Box tracking ──────────────────────────────────────────────────────
        self.box_est_x = -3.0
        self.box_est_y = 1.6
        self.box_est_yaw = 0.0        # 0 = original, grows as box rotates
        self.box_conf = 0.0           # 0..1 confidence in box estimate
        self.lidar_box = None         # raw LiDAR detection

        # ── LiDAR rotation detection ───────────────────────────────────────────
        self._prev_lidar_bearing = None
        self._prev_lidar_range = None
        self._prev_lidar_width = None
        self._bearing_delta = 0.0
        self._range_delta = 0.0
        self._width_delta = 0.0

        # Accumulated rotation signal (positive = CW rotation detected)
        self._rotation_signal = 0.0

        # ── Phase thresholds ──────────────────────────────────────────────────
        # Robot waypoints
        self.BACKUP_X = -3.55
        self.BOX_LANE_Y = 1.55
        self.BOX_SIDE_Y = 2.65       # robot on +Y side of box
        self.APPROACH_X = -1.55      # beside box corner, ready to push

        # Box target positions
        self.PIT_CENTER_X = -0.8      # center of pit
        self.BOX_TARGET_Y = 1.12      # center of pit in Y

        # Rotation
        self.ROTATE_SIGNAL_TARGET = 1.8   # accumulated bearing change = ~90°
        self.ROTATE_PUSH_STEPS = 80
        self.ROTATE_RELEASE_STEPS = 50
        self.MAX_ROTATE_PULSES = 10

        # Detection
        self.LIDAR_RANGE_MAX = 3.5

        # ── Phase sub-state ───────────────────────────────────────────────────
        self._rotate_pulses = 0
        self._rotate_substep = "push"   # "push" | "release"
        self._stuck_ticks = 0
        self._printed_lidar = False

        # ── Diagnostic ────────────────────────────────────────────────────────
        self._last_logged_phase = None

    # ── Velocity command helpers ────────────────────────────────────────────────

    def _cmd(self, lin_x: float, lin_y: float, ang_z: float) -> None:
        self._vel_cmd = torch.tensor([lin_x, lin_y, ang_z],
                                    device=self.device, dtype=torch.float32).view(1, 3)

    def _stop(self) -> None:
        self._cmd(0.0, 0.0, 0.0)

    # ── Dead reckoning ─────────────────────────────────────────────────────────

    def _update_pose(self, proprio: torch.Tensor) -> None:
        """Integrate base velocity to track robot position in world frame."""
        base_lin = proprio[0, 0:3].cpu().numpy()
        base_ang = proprio[0, 3:6].cpu().numpy()

        vx, vy, _ = base_lin
        yaw_rate = base_ang[2]

        self.est_yaw += yaw_rate * self.dt
        cos_y = math.cos(self.est_yaw)
        sin_y = math.sin(self.est_yaw)

        # Velocity in world frame
        vx_w = cos_y * vx - sin_y * vy
        vy_w = sin_y * vx + cos_y * vy

        self.est_x += vx_w * self.dt
        self.est_y += vy_w * self.dt

    def _fuse_lidar_box(self) -> None:
        """Update box estimate from LiDAR detection with low-pass filter."""
        lb = self.lidar_box
        if lb is None:
            self.box_conf = max(0.0, self.box_conf * 0.99)
            return

        bearing = lb["bearing"]
        range_m = lb["range"]

        # Convert to world position
        world_bearing = self.est_yaw + bearing
        cx = self.est_x + math.cos(world_bearing) * range_m
        cy = self.est_y + math.sin(world_bearing) * range_m

        # Low-pass update
        alpha = 0.15
        if self.box_conf < 0.05:
            self.box_est_x = cx
            self.box_est_y = cy
        else:
            self.box_est_x = (1 - alpha) * self.box_est_x + alpha * cx
            self.box_est_y = (1 - alpha) * self.box_est_y + alpha * cy

        self.box_conf = min(1.0, self.box_conf + 0.08)

    def _update_rotation_signal(self) -> None:
        """Monitor LiDAR bearing delta to detect box rotation.

        When the robot pushes the box at an offset from its center, the box
        rotates. The LiDAR bearing to the box changes monotonically during
        rotation, giving us a cheap rotation sensor.
        """
        lb = self.lidar_box
        if lb is None:
            return

        bearing = lb["bearing"]
        range_m = lb["range"]
        width = lb.get("angular_width", 0.3)

        if self._prev_lidar_bearing is not None:
            d_bearing = bearing - self._prev_lidar_bearing
            # Wrap to [-pi, pi]
            while d_bearing > math.pi:
                d_bearing -= 2 * math.pi
            while d_bearing < -math.pi:
                d_bearing += 2 * math.pi

            d_range = range_m - self._prev_lidar_range
            d_width = width - self._prev_lidar_width

            # Low-pass filter deltas
            self._bearing_delta = 0.7 * self._bearing_delta + 0.3 * d_bearing
            self._range_delta = 0.7 * self._range_delta + 0.3 * d_range
            self._width_delta = 0.7 * self._width_delta + 0.3 * d_width

            # Rotation signal: positive bearing_delta during CW push = rotation
            # Scale by range so close contacts dominate
            signal = self._bearing_delta * (2.0 / max(0.5, range_m))
            self._rotation_signal += signal

        self._prev_lidar_bearing = bearing
        self._prev_lidar_range = range_m
        self._prev_lidar_width = width

    # ── LiDAR processing ────────────────────────────────────────────────────────

    def _get_lidar_scan(self, obs) -> torch.Tensor | None:
        extero = obs.get("extero")
        if extero is None:
            return None
        scan = extero.to(device=self.device, dtype=torch.float32)
        if scan.ndim == 1:
            scan = scan.view(1, -1)
        elif scan.ndim > 2:
            scan = scan.reshape(scan.shape[0], -1)
        return scan[0]  # (N,)

    def _detect_box_from_lidar(self, obs):
        """Find the most plausible box cluster in the LiDAR scan.

        Returns a dict with:
          - bearing: angle to box center (rad, + = left)
          - range: estimated range to box
          - angular_width: apparent angular width of box
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
        finite = flat.isfinite()
        values = flat[finite]

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

        # Merge wrap-around
        if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][1] == mask.numel() - 1:
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

            # Range estimate from angular width (box ~0.9m wide)
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

    # ── Policy interface ────────────────────────────────────────────────────────

    def _policy_obs(self, proprio: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Build the 45-dim policy observation vector."""
        idx = 0
        idx += 3   # base_lin_vel (unused but must align)
        base_ang_vel = proprio[0, idx:idx + 3]
        idx += 3
        idx += 3   # velocity_commands (injected below)
        idx += 3   # projected_gravity
        idx += 3
        joint_pos = proprio[0, idx:idx + action_dim]
        idx += action_dim
        joint_vel = proprio[0, idx:idx + action_dim]
        idx += action_dim
        actions = proprio[0, idx:idx + action_dim]

        joint_pos_leg = joint_pos[self.leg_joint_indices]
        joint_vel_leg = joint_vel[self.leg_joint_indices]
        actions_leg = actions[self.leg_joint_indices]
        actions_train = actions_leg * self.env_to_train

        vel_cmd = self._vel_cmd.to(dtype=proprio.dtype, device=self.device)

        return torch.cat([
            base_ang_vel * 0.25,
            proprio[0, 9:12],          # projected_gravity
            vel_cmd[0],
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_train,
        ], dim=-1).unsqueeze(0)

    def _run_policy(self, obs, action_dim: int) -> torch.Tensor:
        """Run locomotion policy and return environment-space action."""
        policy_input = self._policy_obs(obs["proprio"].to(self.device), action_dim)
        with torch.inference_mode():
            action_train = self.policy(policy_input)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        # Scale from training to environment space
        action_env = torch.zeros((1, action_dim), device=self.device, dtype=torch.float32)
        leg_action = action_train[:, :self.LEG_ACTION_DIM] * self.train_to_env
        action_env[:, self.leg_joint_indices] = leg_action
        action_env[:, self.arm_joint_indices] = self.arm_default
        return action_env

    # ── Phase actions ─────────────────────────────────────────────────────────

    def _action_backup(self, obs, action_dim: int) -> torch.Tensor:
        self._cmd(-0.8, 0.0, 0.0)
        return self._run_policy(obs, action_dim)

    def _action_move_to_lane(self, obs, action_dim: int) -> torch.Tensor:
        # Move left to align with box lane
        self._cmd(-0.15, 0.55, 0.0)
        return self._run_policy(obs, action_dim)

    def _action_contact(self, obs, action_dim: int) -> torch.Tensor:
        # Creep forward until we feel the box
        self._cmd(0.25, 0.0, 0.0)
        return self._run_policy(obs, action_dim)

    def _action_detach(self, obs, action_dim: int) -> torch.Tensor:
        # Back away from box
        self._cmd(-0.7, 0.0, 0.0)
        return self._run_policy(obs, action_dim)

    def _action_move_to_side(self, obs, action_dim: int) -> torch.Tensor:
        # Move to +Y side of box, then forward beside it
        self._cmd(-0.05, 0.7, 0.0)
        return self._run_policy(obs, action_dim)

    def _action_rotate_push(self, obs, action_dim: int) -> torch.Tensor:
        # Push the box corner to rotate it CW
        # The robot is on the +Y side of the box, pushing from behind
        yaw_err = max(-0.35, min(0.35, -1.5 * self.est_yaw))
        self._cmd(0.45, -1.0, yaw_err)
        return self._run_policy(obs, action_dim)

    def _action_rotate_release(self, obs, action_dim: int) -> torch.Tensor:
        # Back off after a push pulse so the box settles
        yaw_err = max(-0.35, min(0.35, -1.5 * self.est_yaw))
        self._cmd(-0.4, 0.3, yaw_err)
        return self._run_policy(obs, action_dim)

    def _action_cross(self, obs, action_dim: int) -> torch.Tensor:
        # Walk forward to cross the pit
        self._cmd(0.6, 0.0, 0.0)
        return self._run_policy(obs, action_dim)

    def _action_stuck(self, obs, action_dim: int) -> torch.Tensor:
        # Emergency: try to back out
        self._cmd(-0.5, 0.0, 0.5)
        return self._run_policy(obs, action_dim)

    # ── Phase transitions ──────────────────────────────────────────────────────

    def _transition(self) -> None:
        p = self.phase
        s = self.step

        if p == "BACKUP":
            if self.est_x <= self.BACKUP_X:
                self.phase = "TO_BOX_LANE"
                self.step = 0
                self._prev_x = self.est_x

        elif p == "TO_BOX_LANE":
            if self.est_y >= self.BOX_LANE_Y:
                self.phase = "CONTACT"
                self.step = 0
                self._prev_x = self.est_x

        elif p == "CONTACT":
            # Stuck detection: no forward progress for 30 steps
            dx = abs(self.est_x - self._prev_x)
            if dx < 0.002:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x = self.est_x

            if self._stuck_ticks >= 30 or self.est_x >= -2.5 or s >= 100:
                self.phase = "PUSH_FWD"
                self.step = 0
                self._stuck_ticks = 0

        elif p == "PUSH_FWD":
            # Push the box forward until it reaches the pit
            dx = abs(self.est_x - self._prev_x)
            if dx < 0.002:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x = self.est_x

            box_near_pit = (self.box_est_x >= self.PIT_CENTER_X - 0.3 and self.box_conf > 0.1)
            stuck = self._stuck_ticks >= 25

            if box_near_pit or stuck or s >= 160:
                self.phase = "DETACH"
                self.step = 0
                self._prev_x = self.est_x

        elif p == "DETACH":
            # Back away from the box
            if self.est_x <= self._prev_x - 0.5 or s >= 80:
                self.phase = "TO_SIDE"
                self.step = 0
                self._prev_y = self.est_y

        elif p == "TO_SIDE":
            # Move to the +Y side of the box
            if self.est_y >= self.BOX_SIDE_Y or s >= 150:
                self.phase = "ROTATE"
                self.step = 0
                self._rotate_pulses = 0
                self._rotation_signal = 0.0
                self._prev_x = self.est_x

        elif p == "ROTATE":
            # Rotation sub-state machine
            if self._rotate_substep == "push":
                if s >= self.ROTATE_PUSH_STEPS:
                    self._rotate_substep = "release"
                    self.step = 0
            elif self._rotate_substep == "release":
                if s >= self.ROTATE_RELEASE_STEPS:
                    # Check if box is rotated enough
                    if self._rotation_signal >= self.ROTATE_SIGNAL_TARGET:
                        self.phase = "CROSS"
                        self.step = 0
                    else:
                        self._rotate_pulses += 1
                        if self._rotate_pulses >= self.MAX_ROTATE_PULSES:
                            # Force advance even if rotation isn't detected
                            self.phase = "CROSS"
                            self.step = 0
                        else:
                            self._rotate_substep = "push"
                            self.step = 0

        elif p == "CROSS":
            # Cross the pit — done when robot reaches x threshold
            pass

    # ── Main entry point ──────────────────────────────────────────────────────

    def predicts(self, obs, current_score):
        # Print observation keys once
        if not hasattr(self, "_printed_obs"):
            print("OBS KEYS:", list(obs.keys()))
            if "extero" in obs:
                ex = obs["extero"]
                print(f"extero shape={getattr(ex, 'shape', None)}, dtype={getattr(ex, 'dtype', None)}")
            self._printed_obs = True

        # Get dimensions
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        # Update state
        self._update_pose(proprio)
        self._detect_box_from_lidar(obs)
        self._fuse_lidar_box()
        self._update_rotation_signal()
        self._transition()

        # Compute action for current phase
        p = self.phase
        if p == "BACKUP":
            action = self._action_backup(obs, action_dim)
        elif p == "TO_BOX_LANE":
            action = self._action_move_to_lane(obs, action_dim)
        elif p == "CONTACT":
            action = self._action_contact(obs, action_dim)
        elif p == "PUSH_FWD":
            action = self._action_push_fwd(obs, action_dim)
        elif p == "DETACH":
            action = self._action_detach(obs, action_dim)
        elif p == "TO_SIDE":
            action = self._action_move_to_side(obs, action_dim)
        elif p == "ROTATE":
            if self._rotate_substep == "push":
                action = self._action_rotate_push(obs, action_dim)
            else:
                action = self._action_rotate_release(obs, action_dim)
        elif p == "CROSS":
            action = self._action_cross(obs, action_dim)
        else:
            action = self._action_cross(obs, action_dim)

        # Log every 50 steps or on phase change
        if p != self._last_logged_phase or self.step % 50 == 0:
            lb = self.lidar_box
            lidar_str = (f"bearing={lb['bearing']:.2f} range={lb['range']:.2f}"
                        if lb else "none")
            print(
                f"[D] phase={p:<12} step={self.step:>4}  "
                f"pose=({self.est_x:+.2f}, {self.est_y:+.2f}, yaw={self.est_yaw:+.2f})  "
                f"box=({self.box_est_x:+.2f}, {self.box_est_y:+.2f}) conf={self.box_conf:.2f}  "
                f"lidar=[{lidar_str}]  "
                f"rot_sig={self._rotation_signal:+.3f}  "
                f"pulses={self._rotate_pulses}  "
                f"cmd=({self._vel_cmd[0,0].item():+.2f}, {self._vel_cmd[0,1].item():+.2f}, {self._vel_cmd[0,2].item():+.2f})"
            )
            self._last_logged_phase = p

        self.step += 1
        return {"action": action.cpu().tolist()[0], "giveup": False}

    # ── Push forward action (uses lateral correction from LiDAR) ────────────────

    def _action_push_fwd(self, obs, action_dim: int) -> torch.Tensor:
        """Push box forward toward pit, with lateral correction from LiDAR."""
        lb = self.lidar_box
        lin_y = 0.0
        if lb is not None and abs(lb["bearing"]) < 1.2:
            lin_y = 0.2 * lb["bearing"]
        yaw_err = max(-0.35, min(0.35, -1.4 * self.est_yaw))
        self._cmd(0.95, lin_y, yaw_err)
        base = self._run_policy(obs, action_dim)
        return torch.clamp(base, -1.0, 1.0)
