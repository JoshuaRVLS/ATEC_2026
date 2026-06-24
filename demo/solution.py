"""
Task D: Push box into pit, then cross.

World layout:
  - Robot spawns at (-3, 0), facing +X
  - Box at (-3, 1.6), pit at x≈0 (reward zone x ∈ [-0.7, 0.7])
  - Robot must maneuver around box and push it into pit

Sequence (coordinate-based):
  1. BACK      → back up from box for clearance
  2. LEFT      → walk to the upper/off-center corner of the box
  3. ROTATE_BOX→ push one end so the box rotates from ___ toward |
  4. RIGHT_ALIGN → move down alongside the rotated box
  5. BACK_SIDE → back up to Y < box_Y (behind box)
  6. PUSH_PIT  → push rotated box into pit
  7. CROSS     → walk across pit

Transitions use actual robot WORLD POSITION (from dead reckoning).
"""

import os
import math
import torch


PIT_MIN_X = -0.7
PIT_MAX_X = 0.7


class AlgSolution:

    def __init__(self):
        policy_path = os.path.dirname(os.path.abspath(__file__)) + '/policy.pt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
        self.ROTATE_Y = self.BOX_Y + 0.42  # hit the upper end, not the center
        self.ROTATE_X_STOP = -2.25  # stop before sliding the box into the wall
        self.PIT_ALIGN_Y = 0.25
        self.BACK_SIDE_X = -3.5
        self.BACK_SIDE_Y = -0.75
        self.ALIGN_YAW_TOL = 0.15
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
        self.ROTATE_BOX_STEPS = 280
        self.RIGHT_ALIGN_STEPS = 350
        self.BACK_SIDE_STEPS = 600
        self.PUSH_PIT_STEPS = 700
        self.STABILIZE_STEPS = 250
        self.CROSS_STEPS = 500

        # ── LiDAR ────────────────────────────────────────────────────────────
        self.lidar_box = None
        self._prev_lidar_range = None
        self._prev_lidar_bearing = None
        self._range_history = []
        self._bearing_history = []
        self._MAX_RANGE_HISTORY = 20

        # ── Box estimator ───────────────────────────────────────────────────
        self.est_box_x = None
        self.est_box_y = None

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

            idxs = torch.arange(s, e + 1, device=self.device) % n_bins
            cluster_values = horizontal[idxs]
            cluster_values = cluster_values[cluster_values.isfinite()]
            if cluster_values.numel() == 0:
                continue

            cluster_range = cluster_values.median().item()
            if not math.isfinite(cluster_range) or cluster_range <= 0.0:
                continue
            cluster_range = max(0.4, min(6.0, cluster_range))

            angles = (idxs.float() / float(n_bins - 1)) * (2 * math.pi) - math.pi
            weights = deviation[idxs].clamp_min(1e-4)
            sin_mean = (weights * torch.sin(angles)).sum() / weights.sum()
            cos_mean = (weights * torch.cos(angles)).sum() / weights.sum()
            bearing = math.atan2(sin_mean.item(), cos_mean.item())

            range_score = 1.0 / (1.0 + 0.5 * abs(cluster_range - 1.5))
            bearing_score = 1.0 / (1.0 + 0.3 * abs(bearing))
            width_score = math.sqrt(float(width))
            score = width_score * range_score * bearing_score

            if score > best_score:
                best_score = score
                best = (bearing, cluster_range, angular_w, width)

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
    # Box position estimation (from LiDAR + robot pose)
    # ══════════════════════════════════════════════════════════════════════════

    def _estimate_box_world_pos(self) -> tuple[float, float] | None:
        """Estimate box position in WORLD frame using LiDAR + robot pose.

        Also tracks range history to detect when robot passes the box.
        """
        if self.lidar_box is None:
            return None

        bearing = self.lidar_box['bearing']
        rng = self.lidar_box['range']

        # Track range history for pass-by detection
        self._range_history.append(rng)
        self._bearing_history.append(bearing)
        if len(self._range_history) > self._MAX_RANGE_HISTORY:
            self._range_history.pop(0)
        if len(self._bearing_history) > self._MAX_RANGE_HISTORY:
            self._bearing_history.pop(0)
        self._prev_lidar_range = rng
        self._prev_lidar_bearing = bearing

        # Clamp range
        if rng < 0.5 or rng > 5.0:
            return None

        # Box position in robot frame
        box_rx = rng * math.cos(bearing)
        box_ry = rng * math.sin(bearing)

        # Transform to world frame
        cos_y = math.cos(self.est_yaw)
        sin_y = math.sin(self.est_yaw)

        world_dx = cos_y * box_rx - sin_y * box_ry
        world_dy = sin_y * box_rx + cos_y * box_ry

        self.est_box_x = self.est_x + world_dx
        self.est_box_y = self.est_y + world_dy

        return self.est_box_x, self.est_box_y

    def _detected_box_pass(self) -> bool:
        """Detect if robot has passed the box (box is now behind robot).

        When passing a box:
        1. Range suddenly increases (was 1.5m, now 3m+)
        2. Box bearing shifts (was ahead, now to side or behind)
        """
        if len(self._range_history) < 10:
            return False

        recent = self._range_history[-5:]
        older = self._range_history[-10:-5] if len(self._range_history) >= 10 else self._range_history[:-5]

        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)

        recent_bearing = self._bearing_history[-1] if self._bearing_history else 0.0

        # A range jump plus a side/behind bearing is a stronger pass-by signal
        # than range alone, which can fluctuate as the scan latches onto clutter.
        if avg_recent > avg_older + 1.5 and abs(recent_bearing) > 0.45:
            return True

        return False

    def _estimate_box_yaw(self) -> float | None:
        """Estimate box orientation (yaw) using LiDAR edge analysis.

        Method: Use angular extent of LiDAR cluster to estimate box orientation.
        - When box is perpendicular to robot, angular width is LARGE
        - When box is parallel to robot, angular width is SMALL

        The angular_width from our detection gives us the apparent size.
        Combined with range, we can estimate the effective width and infer rotation.

        Reference:
        - "Real-time 3D LiDAR-based Object Pose Estimation for Mobile Manipulation"
        - "Rectangular Box Detection using 2D LiDAR Range Data"
        """
        if self.lidar_box is None:
            return None

        angular_w = self.lidar_box.get('angular_width', 0)
        rng = self.lidar_box.get('range', 2.0)

        if angular_w <= 0 or rng <= 0:
            return None

        # Box physical width is ~0.8m (from env_cfg)
        BOX_WIDTH = 0.8

        # Angular width θ → chord width w = 2 * r * sin(θ/2)
        # For small angles: w ≈ r * θ
        # Apparent angular width tells us which face we're seeing
        apparent_width = rng * angular_w

        if apparent_width > BOX_WIDTH * 1.3:
            # Angular width larger than physical width → we're seeing a longer face
            # This means box is somewhat rotated, presenting its diagonal or side
            yaw_est = 0.0  # Unknown exact angle, but box is rotated
        elif apparent_width < BOX_WIDTH * 0.7:
            # Angular width smaller → we're seeing edge-on or box is rotated away
            yaw_est = math.pi / 4  # ~45 degree rotation
        else:
            # Normal view, box roughly aligned
            yaw_est = 0.0

        return yaw_est

    def _is_box_in_pit(self) -> bool:
        """Check if box is in pit reward zone (x ∈ [-0.7, 0.7])."""
        if self.est_box_x is None:
            return False
        return PIT_MIN_X <= self.est_box_x <= PIT_MAX_X

    # ══════════════════════════════════════════════════════════════════════════
    # State machine (coordinate-based using world position)
    # ══════════════════════════════════════════════════════════════════════════

    def _transition(self) -> None:
        p = self.phase
        s = self.step

        if p == "BACK":
            if self.est_x <= -4.0 or s >= self.BACK_STEPS:
                self.phase = "LEFT"
                self.step = 0

        elif p == "LEFT":
            # Aim above the box centerline so the +X push creates torque.
            if self.est_y >= self.ROTATE_Y or s >= self.LEFT_STEPS:
                self.phase = "ROTATE_BOX"
                self.step = 0

        elif p == "ROTATE_BOX":
            # This is intentionally short/off-center. A centerline push just
            # translates the box; pushing the upper end should rotate it.
            if self.est_x >= self.ROTATE_X_STOP:
                self.phase = "RIGHT_ALIGN"
                self.step = 0
                return
            if s >= self.ROTATE_BOX_STEPS:
                self.phase = "RIGHT_ALIGN"
                self.step = 0
                return

        elif p == "RIGHT_ALIGN":
            # Move down after the corner push, then back up for the final +X shove.
            # LiDAR is noisy here, so use robot pose as the primary guard.
            if self.est_y <= self.PIT_ALIGN_Y:
                self.phase = "BACK_SIDE"
                self.step = 0
            elif s >= self.RIGHT_ALIGN_STEPS:
                self.phase = "BACK_SIDE"
                self.step = 0

        elif p == "BACK_SIDE":
            # First back up to x < -3.0
            # Then strafe right to y < -0.8 (south of box)
            if self.est_x < self.BACK_SIDE_X and self.est_y < self.BACK_SIDE_Y:
                self.phase = "PUSH_PIT"
                self.step = 0
            elif s >= self.BACK_SIDE_STEPS:
                self.phase = "PUSH_PIT"
                self.step = 0

        elif p == "PUSH_PIT":
            # Keep pushing until box is in pit OR box is past robot
            if self._is_box_in_pit():
                self.phase = "STABILIZE"
                self.step = 0
                return
            if self._detected_box_pass():
                self.phase = "STABILIZE"
                self.step = 0
                return
            if s >= self.PUSH_PIT_STEPS:
                self.phase = "STABILIZE"
                self.step = 0
                return

        elif p == "STABILIZE":
            # Wait for robot yaw to stabilize before crossing
            if abs(self.est_yaw) < 0.15 and s >= 50:
                self.phase = "CROSS"
                self.step = 0
            elif s >= self.STABILIZE_STEPS:
                self.phase = "CROSS"
                self.step = 0

        elif p == "CROSS":
            if self.est_x >= self.PIT_X or s >= self.CROSS_STEPS:
                self.phase = "DONE"
                self.step = 0

    # ══════════════════════════════════════════════════════════════════════════
    # Policy interface (mirrors solution_rl.py)
    # ══════════════════════════════════════════════════════════════════════════

    def _set_body_velocity(self, vx: float, vy: float, wz: float) -> None:
        self._vel_x = vx
        self._vel_y = vy
        self._vel_z = wz

    def _set_world_velocity(self, vx: float, vy: float, wz: float) -> None:
        """Convert desired world XY velocity into the policy's body-frame command."""
        cos_y = math.cos(self.est_yaw)
        sin_y = math.sin(self.est_yaw)
        self._vel_x = cos_y * vx + sin_y * vy
        self._vel_y = -sin_y * vx + cos_y * vy
        self._vel_z = wz

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

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        # ── Pose update + sensing ───────────────────────────────────────────
        self._update_pose(proprio)
        lb = self._detect_box_lidar(obs)
        self.lidar_box = lb
        # ── Box world position estimation ─────────────────────────────────
        self._estimate_box_world_pos()
        self._transition()

        p = self.phase

        # ── Velocity command per phase ──────────────────────────────────────
        if p == "BACK":
            self._set_body_velocity(-1.0, 0.0, 0.0)
        elif p == "LEFT":
            # Strafe left to the upper box corner before the rotation push.
            self._set_body_velocity(0.0, 0.75 if self.est_y < self.ROTATE_Y else 0.0, 0.0)
        elif p == "ROTATE_BOX":
            # Push one end of the box. A slight downward bias keeps contact on
            # the corner instead of riding along the side/wall.
            y_err = self.ROTATE_Y - self.est_y
            if abs(self.est_yaw) > 0.08:
                wz = -self.est_yaw * 0.7
            else:
                wz = 0.0
            self._set_body_velocity(0.45, max(-0.35, min(0.25, 1.0 * y_err - 0.10)), wz)
        elif p == "RIGHT_ALIGN":
            # First square the robot back to world +X. If we strafe while yawed,
            # the body-frame command can become forward motion into the wall.
            if abs(self.est_yaw) > self.ALIGN_YAW_TOL:
                self._set_body_velocity(0.0, 0.0, -self.est_yaw * 1.2)
            else:
                self._set_world_velocity(0.0, -0.70, -self.est_yaw * 0.5)
        elif p == "BACK_SIDE":
            # Always do both back AND strafe in this phase
            # Target: x < -3.5, y < -0.8 (south of box)
            vx = -0.65 if self.est_x > self.BACK_SIDE_X else 0.0
            vy = -0.65 if self.est_y > self.BACK_SIDE_Y else 0.0
            self._set_world_velocity(vx, vy, -self.est_yaw * 0.5)
        elif p == "PUSH_PIT":
            # Final shove: keep the robot roughly centered on the lower side of
            # the now-rotated box so contact does not peel off diagonally.
            y_err = self.PIT_ALIGN_Y - self.est_y
            self._set_world_velocity(0.65, max(-0.30, min(0.30, 0.8 * y_err)), -self.est_yaw * 0.4)
        elif p == "STABILIZE":
            # Stop and correct yaw until stable
            if abs(self.est_yaw) > 0.1:
                self._set_body_velocity(0.0, 0.0, -self.est_yaw * 0.5)
            else:
                self._set_body_velocity(0.0, 0.0, 0.0)
        elif p == "CROSS":
            # Cross pit with yaw correction
            if abs(self.est_yaw) > 0.1:
                wz = -self.est_yaw * 0.5
            else:
                wz = 0.0
            self._set_world_velocity(0.8, 0.0, wz)
        elif p == "DONE":
            self._set_body_velocity(0.0, 0.0, 0.0)

        action = self._run_policy(obs, action_dim)

        # ── Log every 25 steps ──────────────────────────────────────────────
        if self.step % 25 == 0:
            rx = self.est_x
            ry = self.est_y
            ryaw = math.degrees(self.est_yaw)
            bx = f"{self.est_box_x:+.1f}" if self.est_box_x is not None else "---"
            by = f"{self.est_box_y:+.1f}" if self.est_box_y is not None else "---"
            pit = "Y" if self._is_box_in_pit() else " "
            passed = "Y" if self._detected_box_pass() else " "
            brg = f"{math.degrees(self.lidar_box['bearing']):+.0f}deg" if self.lidar_box else "---"
            rng = f"{self.lidar_box['range']:.1f}m" if self.lidar_box else "---"
            print(
                f"[D]{p:<10}|{self.step:<4}|robot=({rx:+.1f},{ry:+.1f},{ryaw:+.0f}deg)|"
                f"box=({bx},{by})|lidar@brg={brg} rng={rng}|pit={pit}pass={passed}"
            )

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}
