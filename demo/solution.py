"""
Task D: Push box into pit, rotate box 90° first.

Sequence:
  1. WALK_TO_BOX   → Walk toward box (forward + left) until near box
  2. PUSH_AND_ROTATE → Push box from +Y side → creates torque → rotates ~90°
  3. PUSH_TO_PIT   → Push rotated box so corner bridges pit
  4. CROSS         → Walk across pit using box corner as bridge

Velocity command format (heading_command=True in policy):
  vel_x = forward speed (m/s)
  vel_y = heading direction (radians, -π to +π)
  vel_z = turn rate (rad/s)

For heading=0: robot faces +X (forward toward pit)
For heading=+1.57: robot faces +Y (left)
For heading=-1.57: robot faces -Y (right)

Pose estimation: Dead reckoning from base_lin_vel + LiDAR yaw correction.
Box tracking: LiDAR triangulation (no GT).
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

        # ── Known initial positions ───────────────────────────────────────────
        # Robot spawns at (-3, 0), box at (-3, 1.6), pit at x≈-0.2
        # Robot's initial heading: facing +X (toward pit)
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0   # 0 = facing +X (toward pit)
        self._yaw_correction = 0.0
        self._wall_bearing_ref = None

        # ── Box pose (LiDAR triangulation) ────────────────────────────────────
        self.lidar_box = None
        self.box_x = -3.0
        self.box_y = 1.6
        self.box_conf = 1.0  # start from known init

        # ── Rotation signal ────────────────────────────────────────────────────
        self._rotation_signal = 0.0
        self._prev_lidar_bearing = None
        self._bearing_delta_smoothed = 0.0

        # ── Velocity command (heading-based) ───────────────────────────────────
        # vel_x = forward speed, vel_y = heading (radians), vel_z = turn rate
        self._vel_x = 0.5
        self._vel_y = 0.0
        self._vel_z = 0.0

        # ── State machine ───────────────────────────────────────────────────────
        # Robot starts at (-3, 0) facing +X.
        # Box at (-3, 1.6). Pit at x≈-0.2.
        # Sequence: approach box → push and rotate → push to pit → cross
        self.phase = "WALK_TO_BOX"
        self.step = 0

        # ── Phase targets ──────────────────────────────────────────────────────
        self.BOX_X_APPROACH = -3.0  # robot near this X = close to box
        self.PIT_X = -0.5           # robot reaches this X = at pit
        self.CROSS_X = 1.0          # robot reaches this X = across pit

        # Rotation detection
        self.ROT_SIG_TARGET = 0.8   # accumulated bearing delta threshold
        self.ROT_PUSH_STEPS = 120    # push for 2.4s
        self.ROT_RELEASE_STEPS = 60  # release for 1.2s
        self.MAX_ROT_CYCLES = 12

        # ── Rotation sub-state ─────────────────────────────────────────────────
        self._rot_cycles = 0
        self._rot_sub = "push"

        # ── Diagnostic ────────────────────────────────────────────────────────
        self._last_phase = None
        self._printed_obs = False
        self._lidar_first_printed = False

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
    # Pose estimation
    # ══════════════════════════════════════════════════════════════════════════

    def _update_pose(self, proprio: torch.Tensor) -> None:
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
    # State machine (coordinate-based)
    # ══════════════════════════════════════════════════════════════════════════

    def _transition(self) -> None:
        p = self.phase
        rx, ry = self.est_x, self.est_y
        lb = self.lidar_box

        if p == "WALK_TO_BOX":
            # Walk forward toward box. Transition when LiDAR detects close box contact.
            if lb is not None and lb["range"] < 0.8:
                # Box is close — we've made contact
                self.phase = "ROTATE_BOX"
                self.step = 0
                self._rot_cycles = 0
                self._rot_sub = "push"
                self._rotation_signal = 0.0
                self._prev_lidar_bearing = None
                self._bearing_delta_smoothed = 0.0
            elif self.step >= 600:
                # Fallback: force advance after max steps
                self.phase = "ROTATE_BOX"
                self.step = 0

        elif p == "ROTATE_BOX":
            if self._rot_sub == "push":
                if self.step >= self.ROT_PUSH_STEPS:
                    self._rot_sub = "release"
                    self.step = 0
            elif self._rot_sub == "release":
                if self.step >= self.ROT_RELEASE_STEPS:
                    self._rot_cycles += 1
                    if self._rotation_signal >= self.ROT_SIG_TARGET:
                        self.phase = "PUSH_TO_PIT"
                        self.step = 0
                    elif self._rot_cycles >= self.MAX_ROT_CYCLES:
                        self.phase = "PUSH_TO_PIT"
                        self.step = 0
                    else:
                        self._rot_sub = "push"
                        self.step = 0

        elif p == "PUSH_TO_PIT":
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

        # ── Velocity command per phase ───────────────────────────────────────
        # heading=0 → face +X (toward pit)
        # heading=+1.57 → face +Y (left)
        # heading=-1.57 → face -Y (right)
        if p == "WALK_TO_BOX":
            # Walk toward box at (-3, 1.6): forward + face left
            self._vel_x = 0.5
            self._vel_y = 0.0  # heading=0 = face +X
            self._vel_z = 0.0
        elif p == "ROTATE_BOX":
            # Push from +Y side to create rotation torque
            # Move forward (into box), heading=0 = face +X
            if self._rot_sub == "push":
                self._vel_x = 0.4
                self._vel_y = 0.0
                self._vel_z = 0.0
            else:
                # Release: slow down
                self._vel_x = 0.1
                self._vel_y = 0.0
                self._vel_z = 0.0
        elif p == "PUSH_TO_PIT":
            # Push box toward pit: move forward
            self._vel_x = 0.5
            self._vel_y = 0.0
            self._vel_z = 0.0
        elif p == "CROSS":
            # Walk across pit
            self._vel_x = 0.4
            self._vel_y = 0.0
            self._vel_z = 0.0

        action = self._run_policy(obs, action_dim)

        # ── Log ─────────────────────────────────────────────────────────────
        if p != self._last_phase:
            lb_str = (f"bear={lb['bearing']:+.2f} rng={lb['range']:.2f} aw={lb['angular_width']:.2f}"
                     if lb else "none")
            rot_sub_str = f" ({self._rot_sub})" if p == "ROTATE_BOX" else ""
            print(
                f"[D] phase={p:<14}{rot_sub_str} step={self.step:>3}  "
                f"robot=({self.est_x:+.2f},{self.est_y:+.2f},yaw={self.est_yaw:+.2f})  "
                f"box=({self.box_x:+.2f},{self.box_y:+.2f}) conf={self.box_conf:.2f}  "
                f"lidar=[{lb_str}]  rot={self._rotation_signal:+.3f}  "
                f"cmd=({self._vel_x:+.1f},h={self._vel_y:+.2f})"
            )
            self._last_phase = p

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}