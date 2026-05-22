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
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)

        # Fixed zero base velocity command for policy input.
        self.fixed_velocity_commands = torch.tensor(
            [0.5, 0.0, 0.0],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        # Task D heuristic:
        # 1. Back up from the box lane.
        # 2. Move left (+Y) until aligned behind the fixed box.
        # 3. Move forward (+X) to push the box into the gap/scoring range.
        self.phase = "BACK_UP"
        self.step = 0 

        self.dt = 0.02
        self.est_x = -3.0
        self.est_y = 0.0
        self.est_yaw = 0.0
        self.BACK_UP_TARGET_X = -3.55
        self.BOX_LANE_Y = 1.55
        self.CONTACT_TARGET_X = -3.72
        self.SIDE_PUSH_START_X = -1.55
        self.BOX_PRE_ROTATE_TARGET_X = -1.55
        self.BOX_INSERT_TARGET_Y = 1.12
        self.BOX_INSERT_Y_TOL = 0.08
        self.CENTER_BOX_Y_TOL = 0.12
        self.CENTER_BOX_MIN_STEPS = 170
        self.CENTER_BOX_MAX_STEPS = 220
        self.BOX_EST_Y_MIN = 0.65
        self.BOX_EST_Y_MAX = 2.05
        self.BOX_ROTATE_TARGET_YAW = -1.35
        self.DETACH_BACKUP_DISTANCE = 0.45
        self.detach_start_x = None
        self.contact_ticks = 0
        self.BOX_LEFT_SIDE_Y = 2.65
        self.ROTATE_CORNER_X = -1.45
        self.ROTATE_RIGHT_TARGET_Y = 1.62
        self.ROTATE_BOX_MIN_STEPS = 260
        self.ROTATE_BOX_MAX_STEPS = 520
        self.ROTATE_PUSH_PULSE_STEPS = 70
        self.ROTATE_RELEASE_OBSERVE_STEPS = 45
        self.MAX_ROTATE_PULSES = 8
        self.SIDE_FORWARD_TARGET_X = -0.85
        self.INSERT_BOX_TARGET_Y = self.BOX_INSERT_TARGET_Y
        self.INSERT_BOX_MIN_STEPS = 220
        self.INSERT_BOX_MAX_STEPS = 620
        self.ALIGN_BEHIND_BOX_MIN_STEPS = 120
        self.ALIGN_BEHIND_BOX_MAX_STEPS = 300
        self.POST_INSERT_BACKUP_STEPS = 90
        self.POST_INSERT_BACKUP_MAX_STEPS = 240
        self.PIT_GUARD_X = -1.05
        self.PIT_RETREAT_X = -1.35
        self.INSERT_MAX_ROBOT_X = -1.02
        self.RELEASE_SAFE_X = -1.25
        self.HEAD_CAMERA_FOV_X_RAD = 0.82
        self.stuck_ticks = 0
        self.prev_est_x = self.est_x
        self.prev_est_y = self.est_y
        self.box_est_x = -3.0
        self.box_est_y = 1.6
        self.box_est_yaw = 0.0
        self.box_yaw_confidence = 0.0
        self.box_sensor_confidence = 0.0
        self.bridge_ready = False
        self.bridge_ready_ticks = 0
        self.bridge_not_ready_ticks = 0
        self.center_box_done = False
        self.insert_attempts = 0
        self.MAX_INSERT_ATTEMPTS = 4
        self.current_score = 0.0
        self.depth_box = None
        self.depth_debug = None
        self.lidar_box = None
        self.prev_lidar_bearing = None
        self.prev_lidar_range = None
        self.prev_lidar_angular_width = None
        self.lidar_bearing_delta = 0.0
        self.lidar_range_delta = 0.0
        self.lidar_width_delta = 0.0
        self.rotate_no_progress_ticks = 0
        self.rotate_release_ticks = 0
        self.rotate_pulse_count = 0
        self.rotate_strategy_index = 0
        self.rotate_strategy_names = ("LEFT_CORNER_CW", "RIGHT_CORNER_CCW", "SIDE_SWEEP")
        self.rotate_last_confidence = 0.0
        self.rotate_stagnant_pulses = 0
        self.rotation_session_active = False
        self._printed_lidar_shape = False
        self.LIDAR_CONTROL_SIGN = 1.0
        self._last_logged_phase = None


    def _resolve_joint_ids(self, candidates: tuple[list[str], ...]) -> list[int]:
        last_error = None
        for names in candidates:
            try:
                ids, found_names = self.robot.find_joints(names)
            except ValueError as err:
                last_error = err
                continue
            if len(ids) == len(names):
                if candidates is self.ARM_JOINT_NAME_CANDIDATES:
                    self.arm_joint_names = list(found_names)
                return list(ids)
        raise ValueError(
            f"Cannot resolve required joints from candidates: {candidates}. Last error: {last_error}"
        )

    def _resolve_ee_body_name(self) -> str:
        last_error = None
        for name in self.EE_BODY_NAME_CANDIDATES:
            try:
                body_ids, _ = self.robot.find_bodies(name)
            except ValueError as err:
                last_error = err
                continue
            if len(body_ids) == 1:
                return name
        raise ValueError(
            f"Cannot resolve EE body from candidates: {self.EE_BODY_NAME_CANDIDATES}. Last error: {last_error}"
        )

    def _ensure_cartesian_targets(self):
        self.cartesian_ctrl.reset()

    def _compute_arm_overlay_action(self) -> torch.Tensor:
        self._ensure_cartesian_targets()

        arm_jpos_des = self.cartesian_ctrl.compute_base(
            self.ee_pos_target_b,
            self.ee_quat_target_b,
        )

        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self.arm_ids] = arm_jpos_des
        full_target[:, self.gripper_ids] = self.gripper_open_pos.repeat(full_target.shape[0], 1)

        return (full_target - self.default_joint_pos) / self.ACTION_SCALE

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Return fixed velocity commands for policy input."""
        num_envs = proprio.shape[0]

        cmd = self.fixed_velocity_commands.to(dtype=proprio.dtype, device=self.device)
        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)

        expected_dim = 3 + 3 + 3 + 3 + action_dim + action_dim + action_dim

        idx = 0
        _base_lin_vel = proprio[:, idx:idx + 3]
        idx += 3

        base_ang_vel = proprio[:, idx:idx + 3]
        idx += 3

        _velocity_commands_env = proprio[:, idx:idx + 3]
        idx += 3

        projected_gravity = proprio[:, idx:idx + 3]
        idx += 3

        joint_pos_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        joint_vel_all = proprio[:, idx:idx + action_dim]
        idx += action_dim

        actions_all = proprio[:, idx:idx + action_dim]

        joint_pos_leg = joint_pos_all[:, self.leg_joint_indices]
        joint_vel_leg = joint_vel_all[:, self.leg_joint_indices]
        actions_env_leg = actions_all[:, self.leg_joint_indices]

        actions_train_leg = actions_env_leg * self.env_to_train_action_scale.to(dtype=proprio.dtype)
        velocity_commands = self._get_velocity_commands(proprio)

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                velocity_commands,
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

        return policy_obs

    def _map_policy_action_to_env_action(self, action_train: torch.Tensor, action_dim: int) -> torch.Tensor:
        """Map training-time 12D leg action to current env 20D full-body action."""
        if action_train.shape[-1] != self.leg_action_dim:
            raise ValueError(
                f"Policy output dim mismatch: got {action_train.shape[-1]}, expected {self.leg_action_dim}"
            )

        num_envs = action_train.shape[0]
        leg_action_env = action_train * self.train_to_env_action_scale

        action_env = torch.zeros(
            (num_envs, action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        action_env[:, self.leg_joint_indices] = leg_action_env
        action_env[:, self.arm_joint_indices] = self.arm_default_action.repeat(num_envs, 1)

        return action_env

    def _compute_base_action(self, obs, action_dim: int) -> torch.Tensor:
        """Run the locomotion baseline and map it into the current env action space."""
        policy_obs = self._extract_policy_obs(obs, action_dim)

        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)

        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        return self._map_policy_action_to_env_action(action_train, action_dim)

    def _set_velocity_command(self, lin_x: float, lin_y: float, ang_z: float) -> None:
        if not self.bridge_ready and self.est_x > self.PIT_GUARD_X:
            lin_x = min(lin_x, -0.35)
        if self.phase == "INSERT_BOX_TO_HOLE" and self.est_x > self.INSERT_MAX_ROBOT_X:
            lin_x = min(lin_x, -0.15)
        self.fixed_velocity_commands = torch.tensor(
            [lin_x, lin_y, ang_z], device=self.device, dtype=torch.float32
        ).view(1, 3)

    def _log_phase_command(self) -> None:
        if self.phase != self._last_logged_phase or self.step % 50 == 0:
            cmd = self.fixed_velocity_commands.detach().cpu().view(-1).tolist()
            print(
                f"[TaskD] phase={self.phase} step={self.step} "
                f"cmd=({cmd[0]:+.2f}, {cmd[1]:+.2f}, {cmd[2]:+.2f}) "
                f"pose=({self.est_x:+.2f}, {self.est_y:+.2f}, {self.est_yaw:+.2f}) "
                f"box_est=({self.box_est_x:+.2f}, {self.box_est_y:+.2f}, {self.box_est_yaw:+.2f}) "
                f"rot_x_target={self._rotate_x_target():+.2f} "
                f"yaw_conf={self.box_yaw_confidence:.2f} sensor_conf={self.box_sensor_confidence:.2f} "
                f"bridge_ready={self.bridge_ready} insert_attempts={self.insert_attempts} "
                f"rotate_pulses={self.rotate_pulse_count} rotate_strategy={self._rotate_strategy_name()} "
                f"depth_box={self.depth_box} depth_debug={self.depth_debug} lidar_box={self.lidar_box}"
            )
            self._last_logged_phase = self.phase

    def _update_pose_estimate(self, proprio: torch.Tensor) -> None:
        """Dead-reckon approximate robot XY from proprio velocity feedback."""
        base_lin_vel = proprio[0, 0:3]
        base_ang_vel = proprio[0, 3:6]

        vx_body = float(base_lin_vel[0].item())
        vy_body = float(base_lin_vel[1].item())
        yaw_rate = float(base_ang_vel[2].item())

        self.est_yaw += yaw_rate * self.dt
        cos_yaw = math.cos(self.est_yaw)
        sin_yaw = math.sin(self.est_yaw)

        vx_world = cos_yaw * vx_body - sin_yaw * vy_body
        vy_world = sin_yaw * vx_body + cos_yaw * vy_body

        self.est_x += vx_world * self.dt
        self.est_y += vy_world * self.dt

    def _update_stuck_counter(self) -> None:
        dx = abs(self.est_x - self.prev_est_x)
        dy = abs(self.est_y - self.prev_est_y)
        if self.phase == "PUSH_BOX" and dx < 0.002 and dy < 0.002:
            self.stuck_ticks += 1
        else:
            self.stuck_ticks = 0
        if self.phase == "CONTACT_BOX" and dx < 0.002 and dy < 0.002:
            self.contact_ticks += 1
        else:
            self.contact_ticks = 0
        self.prev_est_x = self.est_x
        self.prev_est_y = self.est_y

    def _fuse_box_position_candidate(
        self,
        cand_x: float,
        cand_y: float,
        confidence: float,
        max_jump: float = 1.2,
    ) -> bool:
        """Fuse a sensed box position only when it is plausible enough."""
        if confidence <= 0.0:
            return False

        jump = math.hypot(cand_x - self.box_est_x, cand_y - self.box_est_y)
        if jump > max_jump:
            return False

        alpha = float(max(0.04, min(0.35, confidence)))
        self.box_est_x = (1.0 - alpha) * self.box_est_x + alpha * cand_x
        self.box_est_y = (1.0 - alpha) * self.box_est_y + alpha * cand_y
        self.box_sensor_confidence = min(1.0, 0.92 * self.box_sensor_confidence + confidence)
        return True

    def _fuse_depth_box_position(self) -> None:
        """Use camera depth as a high-confidence box center when available."""
        if self.depth_box is None:
            return
        area = self.depth_box.get("area_frac", 0.0)
        width = self.depth_box.get("width_frac", 0.0)
        distance = self.depth_box.get("distance", 0.0)
        if not (0.01 <= area <= 0.35 and 0.04 <= width <= 0.65 and 0.25 <= distance <= 4.0):
            return

        bearing = float(self.depth_box["x_error"]) * (self.HEAD_CAMERA_FOV_X_RAD * 0.5)
        world_bearing = self.est_yaw + bearing
        cand_x = self.est_x + math.cos(world_bearing) * distance
        cand_y = self.est_y + math.sin(world_bearing) * distance
        confidence = 0.22 + min(0.16, area * 0.8)
        self._fuse_box_position_candidate(cand_x, cand_y, confidence, max_jump=1.0)

    def _update_bridge_ready(self) -> None:
        """Debounce bridge readiness so one noisy sensor frame cannot flip it."""
        raw_ready = self._bridge_pose_ready_raw()
        if raw_ready:
            self.bridge_ready_ticks += 1
            self.bridge_not_ready_ticks = 0
        else:
            self.bridge_not_ready_ticks += 1
            self.bridge_ready_ticks = 0

        if self.bridge_ready_ticks >= 6:
            self.bridge_ready = True
        elif self.bridge_not_ready_ticks >= 10:
            self.bridge_ready = False

    def _update_box_pose_model(self) -> None:
        """Maintain a coarse box pose estimate for target-based manipulation.

        We do not get ground-truth box pose in obs. This estimate combines contact
        assumptions with LiDAR bearing changes so phase decisions can target the
        box pose instead of relying only on fixed step counts.
        """
        allow_lidar_position_update = (
            not self.center_box_done
            and self.box_yaw_confidence < 0.35
            and self.phase in ("MOVE_LEFT_TO_BOX_LANE", "CONTACT_BOX")
        )
        if (
            allow_lidar_position_update
            and self.lidar_box is not None
            and self.lidar_box["range"] <= 2.6
            and self.lidar_box["count"] >= 12
        ):
            bearing = self.lidar_box["bearing"]
            # Avoid fusing side/back wall-like clusters as box position. During
            # corner/behind alignment, LiDAR often sees the box from the side.
            if abs(bearing) < 1.10 or self.phase in ("MOVE_LEFT_TO_BOX_LANE", "CONTACT_BOX", "PUSH_BOX"):
                bearing_world = self.est_yaw + bearing
                range_proxy = self.lidar_box.get("range", 1.2)
                lidar_x = self.est_x + math.cos(bearing_world) * range_proxy
                lidar_y = self.est_y + math.sin(bearing_world) * range_proxy
                self._fuse_box_position_candidate(lidar_x, lidar_y, confidence=0.12, max_jump=1.4)

        self._fuse_depth_box_position()

        if self.phase in ("CONTACT_BOX", "PUSH_BOX"):
            observed_x = self.est_x + 0.72
            observed_y = self.est_y
            self.box_est_x = max(self.box_est_x, min(observed_x, self.BOX_PRE_ROTATE_TARGET_X + 0.18))
            self.box_est_y = 0.92 * self.box_est_y + 0.08 * observed_y

        elif self.phase == "MOVE_FORWARD_BESIDE_BOX" and not self.center_box_done:
            self.box_est_y = 0.96 * self.box_est_y + 0.04 * (self.est_y - 0.85)

        elif self.phase == "CENTER_BOX_Y":
            # While pushing from the +Y side, model the box sliding toward the
            # hole center. This keeps the next rotation away from the pit edge.
            y_error = self.box_est_y - self.BOX_INSERT_TARGET_Y
            if y_error > self.CENTER_BOX_Y_TOL:
                self.box_est_y -= 0.006
            elif y_error < -self.CENTER_BOX_Y_TOL:
                self.box_est_y += 0.002

        elif self.phase == "ROTATE_BOX_RIGHT":
            yaw_before = self.box_est_yaw
            sensor_rotation = self._lidar_rotation_progress()
            if sensor_rotation > 0.006:
                yaw_step = min(0.030, 0.004 + sensor_rotation)
                self.box_est_yaw = max(self.BOX_ROTATE_TARGET_YAW, self.box_est_yaw - yaw_step)

            if abs(self.box_est_yaw - yaw_before) < 0.0015:
                self.rotate_no_progress_ticks += 1
            else:
                self.rotate_no_progress_ticks = 0
                self.box_yaw_confidence = min(1.0, self.box_yaw_confidence + 0.05)

        elif self.phase == "INSERT_BOX_TO_HOLE":
            y_error = self.box_est_y - self.BOX_INSERT_TARGET_Y
            self.box_est_y -= max(-0.0035, min(0.0045, 0.004 * y_error))

        self.box_est_y = float(max(self.BOX_EST_Y_MIN, min(self.BOX_EST_Y_MAX, self.box_est_y)))

        self.box_sensor_confidence = max(0.0, self.box_sensor_confidence * 0.995)
        self._update_bridge_ready()

    def _box_x_ready_for_rotation(self) -> bool:
        return self.box_est_x >= self.BOX_PRE_ROTATE_TARGET_X or self.est_x >= self.SIDE_PUSH_START_X

    def _box_y_centered_for_rotation(self) -> bool:
        return abs(self.box_est_y - self.BOX_INSERT_TARGET_Y) <= self.CENTER_BOX_Y_TOL

    def _box_rotation_ready(self) -> bool:
        yaw_ready = self.box_est_yaw <= -0.72
        confidence_ready = self.box_yaw_confidence >= 0.35
        enough_contact = self._rotate_elapsed_steps() >= self.ROTATE_PUSH_PULSE_STEPS * 2
        return yaw_ready and confidence_ready and enough_contact

    def _at_rotate_lane(self) -> bool:
        """Robot must be on the +Y/left side before applying clockwise box rotation."""
        return self.est_y >= self.BOX_LEFT_SIDE_Y - 0.25

    def _rotate_x_target(self) -> float:
        """Target x beside the box corner, clamped before the pit."""
        return min(self.PIT_GUARD_X - 0.22, self.box_est_x + 0.20)

    def _at_rotate_x_position(self) -> bool:
        return abs(self.est_x - self._rotate_x_target()) <= 0.12

    def _rotate_elapsed_steps(self) -> int:
        current_pulse_steps = self.step if self.phase == "ROTATE_BOX_RIGHT" else 0
        return self.rotate_pulse_count * self.ROTATE_PUSH_PULSE_STEPS + current_pulse_steps

    def _box_rotation_progress_seen(self) -> bool:
        return self._lidar_rotation_progress() > 0.002 or self.box_est_yaw <= -0.80

    def _rotate_strategy_name(self) -> str:
        return self.rotate_strategy_names[self.rotate_strategy_index % len(self.rotate_strategy_names)]

    def _advance_rotate_strategy(self) -> None:
        self.rotate_strategy_index = (self.rotate_strategy_index + 1) % len(self.rotate_strategy_names)
        self.rotate_stagnant_pulses = 0

    def _update_rotate_strategy_after_pulse(self) -> None:
        confidence_gain = self.box_yaw_confidence - self.rotate_last_confidence
        if confidence_gain < 0.03:
            self.rotate_stagnant_pulses += 1
        else:
            self.rotate_stagnant_pulses = 0
        self.rotate_last_confidence = self.box_yaw_confidence
        if self.rotate_stagnant_pulses >= 2:
            self._advance_rotate_strategy()

    def _lidar_rotation_progress(self) -> float:
        """Sensor cue that the box is rotating under diagonal contact."""
        if self.lidar_box is None:
            return 0.0
        bearing_cue = max(0.0, -self.lidar_bearing_delta)
        width_cue = max(0.0, self.lidar_width_delta)
        range_cue = max(0.0, -self.lidar_range_delta)
        return 0.85 * bearing_cue + 0.20 * width_cue + 0.04 * range_cue

    def _rotation_contact_stalled(self) -> bool:
        return self.phase == "ROTATE_BOX_RIGHT" and self.rotate_no_progress_ticks > 85

    def _box_inserted_enough(self) -> bool:
        pose_ready = abs(self.box_est_y - self.BOX_INSERT_TARGET_Y) <= self.BOX_INSERT_Y_TOL
        sensor_ready = self._box_is_not_centered_front()
        robot_still_safe = self.est_x <= self.INSERT_MAX_ROBOT_X + 0.05
        return self.step >= self.INSERT_BOX_MIN_STEPS and pose_ready and sensor_ready and robot_still_safe

    def _bridge_pose_ready_raw(self) -> bool:
        """Only allow crossing when the estimated box pose can plausibly bridge the hole."""
        y_ready = abs(self.box_est_y - self.BOX_INSERT_TARGET_Y) <= self.BOX_INSERT_Y_TOL
        yaw_ready = self.box_est_yaw <= -0.75
        confidence_ready = self.box_yaw_confidence >= 0.35
        x_ready = self.box_est_x >= -0.85
        return y_ready and yaw_ready and confidence_ready and x_ready

    def _bridge_pose_ready(self) -> bool:
        return self.bridge_ready

    def _get_image_tensor(self, obs, *names):
        image_obs = obs.get("image", {}) if isinstance(obs, dict) else {}
        if not isinstance(image_obs, dict):
            return None
        for name in names:
            value = image_obs.get(name)
            if value is not None:
                return value
        return None

    def _get_depth_image(self, obs):
        # For Task D the head depth often sees a broad flat platform at almost
        # constant depth. Prefer the end-effector camera, then fall back.
        self.depth_source = None
        depth = None
        image_obs = obs.get("image", {}) if isinstance(obs, dict) else {}
        if isinstance(image_obs, dict):
            for name in ("ee_depth", "head_depth", "video_depth"):
                value = image_obs.get(name)
                if value is not None:
                    depth = value
                    self.depth_source = name
                    break
        if depth is None:
            return None

        depth = depth.to(device=self.device, dtype=torch.float32)
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2:
            return None
        return depth

    def _estimate_box_from_depth(self, obs):
        """Estimate a plausible box bbox from head depth."""
        depth = self._get_depth_image(obs)
        if depth is None:
            self.depth_box = None
            self.depth_debug = "no_depth"
            return None

        height, width = depth.shape
        row0, row1 = int(height * 0.22), int(height * 0.92)
        col0, col1 = int(width * 0.04), int(width * 0.96)
        roi = depth[row0:row1, col0:col1]

        valid = torch.isfinite(roi) & (roi > 0.15) & (roi < 5.0)
        valid_depth = roi[valid]
        if valid_depth.numel() < 200:
            self.depth_box = None
            self.depth_debug = f"{self.depth_source}:few_valid:{int(valid_depth.numel())}"
            return None

        flat = valid_depth.flatten()
        q08 = torch.kthvalue(flat, max(1, int(flat.numel() * 0.08))).values
        q25 = torch.kthvalue(flat, max(1, int(flat.numel() * 0.25))).values
        near_depth = torch.minimum(q08 + 0.55, q25 + 0.25)
        near_mask = valid & (roi <= near_depth)

        # Remove isolated rows/cols and keep the largest dense rectangular blob.
        for _ in range(2):
            row_counts = near_mask.sum(dim=1)
            col_counts = near_mask.sum(dim=0)
            row_keep = row_counts > max(4, int(0.012 * near_mask.shape[1]))
            col_keep = col_counts > max(4, int(0.012 * near_mask.shape[0]))
            near_mask = near_mask & row_keep[:, None] & col_keep[None, :]

        near_pixels = int(near_mask.sum().item())
        roi_pixels = int(roi.numel())
        area_frac = float(near_pixels) / float(max(1, roi_pixels))
        if near_pixels < 80 or area_frac > 0.65:
            self.depth_box = None
            self.depth_debug = (
                f"{self.depth_source}:bad_area:pixels={near_pixels},"
                f"area={area_frac:.3f},near={float(near_depth.item()):.2f},"
                f"min={float(valid_depth.min().item()):.2f},"
                f"max={float(valid_depth.max().item()):.2f}"
            )
            return None

        ys_roi, xs_roi = torch.where(near_mask)
        rows = torch.unique(ys_roi)
        cols = torch.unique(xs_roi)
        bbox_h = int(rows[-1].item() - rows[0].item() + 1)
        bbox_w = int(cols[-1].item() - cols[0].item() + 1)
        width_frac = float(bbox_w) / float(max(1, near_mask.shape[1]))
        height_frac = float(bbox_h) / float(max(1, near_mask.shape[0]))
        if width_frac < 0.025 or height_frac < 0.035 or width_frac > 0.95 or height_frac > 0.95:
            self.depth_box = None
            self.depth_debug = f"{self.depth_source}:bad_bbox:w={width_frac:.2f},h={height_frac:.2f}"
            return None

        xs = xs_roi
        xs = xs.to(torch.float32) + float(col0)
        center_x = xs.mean()
        center_y = ys_roi.to(torch.float32).mean() + float(row0)
        image_center_x = torch.tensor(float(width - 1) * 0.5, device=self.device)
        image_center_y = torch.tensor(float(height - 1) * 0.5, device=self.device)
        x_error = ((center_x - image_center_x) / image_center_x).clamp(-1.0, 1.0)
        y_error = ((center_y - image_center_y) / image_center_y).clamp(-1.0, 1.0)
        distance = roi[near_mask].median()

        estimate = {
            "x_error": float(x_error.item()),
            "y_error": float(y_error.item()),
            "distance": float(distance.item()),
            "pixels": near_pixels,
            "area_frac": area_frac,
            "width_frac": width_frac,
            "height_frac": height_frac,
        }
        self.depth_box = estimate
        self.depth_debug = f"{self.depth_source}:ok"
        return estimate

    def _box_centered_from_depth(self, obs) -> bool:
        estimate = self._estimate_box_from_depth(obs)
        if estimate is None:
            return False
        return abs(estimate["x_error"]) < 0.16 and estimate["distance"] < 3.0

    def _depth_corrected_lateral_cmd(self, obs, base_lin_y: float, gain: float = 0.35) -> float:
        return base_lin_y

    def _get_lidar_scan(self, obs):
        """Return the flattened Task D LiDAR observation from obs['extero']."""
        scan = obs.get("extero") if isinstance(obs, dict) else None
        if scan is None:
            return None

        scan = scan.to(device=self.device, dtype=torch.float32)
        if scan.ndim == 1:
            scan = scan.view(1, -1)
        elif scan.ndim > 2:
            scan = scan.reshape(scan.shape[0], -1)
        return scan

    def _extract_lidar_horizontal_profile(self, obs):
        scan = self._get_lidar_scan(obs)
        if scan is None or scan.numel() == 0:
            return None

        flat = scan[0].flatten()
        finite = torch.isfinite(flat)
        valid = flat[finite]
        if valid.numel() < 32:
            return None

        if not self._printed_lidar_shape:
            sample = flat[: min(12, flat.numel())].detach().cpu().tolist()
            print(
                "[TaskD LiDAR] "
                f"shape={tuple(scan.shape)} finite={int(finite.sum().item())}/{flat.numel()} "
                f"min={float(valid.min().item()):+.3f} max={float(valid.max().item()):+.3f} "
                f"mean={float(valid.mean().item()):+.3f} sample={sample}"
            )
            self._printed_lidar_shape = True

        # Collapse vertical channels into horizontal bins. Most ATEC LiDAR patterns
        # are 360 horizontal rays; if that changes, fall back to using all bins.
        if flat.numel() % 360 == 0:
            cols = flat.view(-1, 360)
            col_finite = torch.isfinite(cols)
            safe_cols = torch.where(col_finite, cols, torch.zeros_like(cols))
            counts = col_finite.sum(dim=0).clamp_min(1)
            horizontal = safe_cols.sum(dim=0) / counts
        else:
            horizontal = flat

        return horizontal

    def _find_lidar_clusters(self, mask: torch.Tensor) -> list[tuple[int, int]]:
        """Find contiguous angular clusters in a 1D circular LiDAR mask."""
        indices = torch.where(mask)[0].detach().cpu().tolist()
        if not indices:
            return []

        clusters = []
        start = prev = indices[0]
        for idx in indices[1:]:
            if idx == prev + 1:
                prev = idx
            else:
                clusters.append((start, prev))
                start = prev = idx
        clusters.append((start, prev))

        # Merge wrap-around cluster touching both ends of the circular scan.
        if len(clusters) > 1 and clusters[0][0] == 0 and clusters[-1][1] == mask.numel() - 1:
            first = clusters.pop(0)
            last = clusters.pop(-1)
            clusters.insert(0, (last[0], first[1] + mask.numel()))
        return clusters

    def _estimate_box_from_lidar(self, obs):
        """Estimate box bearing/range from clustered LiDAR outliers.

        This is not full SLAM. It converts the flattened height scan into a circular
        angular profile, clusters non-floor returns, and treats the most plausible
        mid-size cluster as the box.
        """
        horizontal = self._extract_lidar_horizontal_profile(obs)
        if horizontal is None:
            self.lidar_box = None
            return None

        finite_h = torch.isfinite(horizontal)
        values = horizontal[finite_h]
        if values.numel() < 32:
            self.lidar_box = None
            return None

        median = values.median()
        deviation = torch.abs(horizontal - median)
        valid_deviation = deviation[finite_h]
        kth_index = max(1, int(valid_deviation.numel() * 0.90))
        threshold = torch.kthvalue(valid_deviation, kth_index).values.clamp_min(0.05)
        mask = finite_h & (deviation >= threshold)

        if int(mask.sum().item()) < 3:
            self.lidar_box = None
            return None

        num_bins = max(2, horizontal.numel())
        clusters = self._find_lidar_clusters(mask)
        if not clusters:
            self.lidar_box = None
            return None

        best = None
        best_score = -1.0
        for start, end in clusters:
            raw_idx = torch.arange(start, end + 1, device=self.device) % num_bins
            width = raw_idx.numel()
            angular_width = float(width) * (2.0 * math.pi / float(num_bins))
            range_proxy = float(max(0.45, min(5.0, 0.85 / max(angular_width, 0.08))))
            if width < 8 or width > int(0.30 * num_bins):
                continue
            if angular_width < 0.12 or range_proxy > 3.2:
                continue
            weights_i = deviation[raw_idx].clamp_min(1e-4)
            strength = float(weights_i.mean().item())
            center_penalty = abs(((start + end) * 0.5 / float(num_bins - 1)) * (2.0 * math.pi) - math.pi)
            range_penalty = abs(range_proxy - 1.2)
            score = strength * math.sqrt(float(width)) / (1.0 + 0.15 * center_penalty + 0.35 * range_penalty)
            if score > best_score:
                best_score = score
                best = (raw_idx, width, strength, angular_width, range_proxy)

        if best is None:
            self.lidar_box = None
            return None

        idx, width, strength, angular_width, range_proxy = best
        weights = deviation[idx].clamp_min(1e-4)
        angles = (idx.to(torch.float32) / float(num_bins - 1)) * (2.0 * math.pi) - math.pi
        sin_mean = torch.sum(weights * torch.sin(angles)) / torch.sum(weights)
        cos_mean = torch.sum(weights * torch.cos(angles)) / torch.sum(weights)
        bearing = math.atan2(float(sin_mean.item()), float(cos_mean.item()))

        estimate = {
            "bearing": bearing,
            "range": range_proxy,
            "angular_width": angular_width,
            "count": int(width),
            "strength": strength,
        }
        if self.prev_lidar_bearing is not None:
            delta = bearing - self.prev_lidar_bearing
            while delta > math.pi:
                delta -= 2.0 * math.pi
            while delta < -math.pi:
                delta += 2.0 * math.pi
            self.lidar_bearing_delta = 0.85 * self.lidar_bearing_delta + 0.15 * delta
        if self.prev_lidar_range is not None:
            range_delta = range_proxy - self.prev_lidar_range
            self.lidar_range_delta = 0.85 * self.lidar_range_delta + 0.15 * range_delta
        if self.prev_lidar_angular_width is not None:
            width_delta = angular_width - self.prev_lidar_angular_width
            self.lidar_width_delta = 0.85 * self.lidar_width_delta + 0.15 * width_delta
        self.prev_lidar_bearing = bearing
        self.prev_lidar_range = range_proxy
        self.prev_lidar_angular_width = angular_width
        self.lidar_box = estimate
        return estimate

    def _lidar_corrected_lateral_cmd(self, base_lin_y: float, gain: float = 0.20) -> float:
        """Use LiDAR bearing as a small lateral correction toward the box."""
        if self.lidar_box is None:
            return base_lin_y

        bearing = self.lidar_box["bearing"]
        if abs(bearing) > 1.35:
            return base_lin_y

        corrected = base_lin_y + self.LIDAR_CONTROL_SIGN * gain * bearing
        return float(max(-0.65, min(0.65, corrected)))

    def predicts(self, obs, current_score):
        """Run policy inference and return current-env full-body action."""
        # if current_score > 1:
        #     return {'action': [], 'giveup': True}

        if not hasattr(self, "_printed_obs_keys"):
            print("OBS KEYS:", obs.keys())
            if "image" in obs:
                print("IMAGE KEYS:", obs["image"].keys())
                for k, v in obs["image"].items():
                    print(k, type(v), getattr(v, "shape", None), getattr(v, "dtype", None))
            self._printed_obs_keys = True
        
        self.current_score = float(current_score)
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3
        self._update_pose_estimate(proprio)
        self._update_stuck_counter()
        self._estimate_box_from_depth(obs)
        self._estimate_box_from_lidar(obs)
        self._update_box_pose_model()

        if self.phase == "BACK_UP" and self.est_x <= self.BACK_UP_TARGET_X:
            self.phase = "MOVE_LEFT_TO_BOX_LANE"
            self.step = 0
        elif self.phase == "MOVE_LEFT_TO_BOX_LANE" and self.est_y >= self.BOX_LANE_Y:
            self.phase = "CONTACT_BOX"
            self.step = 0
        elif self.phase == "CONTACT_BOX" and (
            self.est_x >= self.CONTACT_TARGET_X
            or self.contact_ticks >= 20
            or self.step >= 70
        ):
            self.phase = "DETACH_FROM_BOX"
            self.detach_start_x = self.est_x
            self.rotation_session_active = False
            self.step = 0
        elif self.phase == "PUSH_BOX" and (self._box_x_ready_for_rotation() or self.stuck_ticks >= 25):
            self.phase = "DETACH_FROM_BOX"
            self.detach_start_x = self.est_x
            self.rotation_session_active = False
            self.center_box_done = False
            self.step = 0
        elif (
            self.phase == "DETACH_FROM_BOX"
            and self.detach_start_x is not None
            and self.est_x <= self.detach_start_x - self.DETACH_BACKUP_DISTANCE
        ):
            self.phase = "MOVE_LEFT_OF_BOX"
            self.detach_start_x = None
            self.step = 0
        elif self.phase == "MOVE_LEFT_OF_BOX" and self.est_y >= self.BOX_LEFT_SIDE_Y:
            self.phase = "MOVE_FORWARD_BESIDE_BOX" if self.center_box_done else "CENTER_BOX_Y"
            self.step = 0
        elif self.phase == "CENTER_BOX_Y" and (
            (self.step >= self.CENTER_BOX_MIN_STEPS and self._box_y_centered_for_rotation())
            or self.step >= self.CENTER_BOX_MAX_STEPS
        ):
            self.center_box_done = True
            self.phase = "MOVE_FORWARD_BESIDE_BOX"
            self.step = 0
        elif self.phase == "MOVE_FORWARD_BESIDE_BOX" and (
            (self._at_rotate_x_position() or self.step >= 360)
            and not self._at_rotate_lane()
        ):
            self.phase = "MOVE_LEFT_OF_BOX"
            self.step = 0
        elif self.phase == "MOVE_FORWARD_BESIDE_BOX" and (
            (self._at_rotate_x_position() or self.step >= 360)
            and self._at_rotate_lane()
            and not self._box_y_centered_for_rotation()
        ):
            self.center_box_done = False
            self.phase = "MOVE_LEFT_OF_BOX"
            self.step = 0
        elif self.phase == "MOVE_FORWARD_BESIDE_BOX" and (
            (self._at_rotate_x_position() or self.step >= 360)
            and self._at_rotate_lane()
            and self._box_y_centered_for_rotation()
        ):
            self.phase = "ROTATE_BOX_RIGHT"
            self.rotate_no_progress_ticks = 0
            self.rotate_release_ticks = 0
            if not self.rotation_session_active:
                self.rotate_pulse_count = 0
                self.box_est_yaw = 0.0
                self.box_yaw_confidence = 0.0
                self.rotate_strategy_index = 0
                self.rotate_last_confidence = 0.0
                self.rotate_stagnant_pulses = 0
                self.rotation_session_active = True
            self.step = 0
        elif self.phase in ("ROTATE_BOX_RIGHT", "ROTATE_RELEASE_OBSERVE", "ALIGN_BEHIND_ROTATED_BOX", "INSERT_BOX_TO_HOLE") and (
            not self.bridge_ready and self.est_x > self.PIT_GUARD_X
        ):
            self.phase = "RETREAT_FROM_PIT"
            self.step = 0
        elif self.phase == "RETREAT_FROM_PIT" and self.est_x <= self.PIT_RETREAT_X:
            if self._box_rotation_ready():
                self.phase = "ALIGN_BEHIND_ROTATED_BOX"
            else:
                self.phase = "MOVE_FORWARD_BESIDE_BOX" if self._at_rotate_lane() else "MOVE_LEFT_OF_BOX"
            self.step = 0
        elif self.phase == "ROTATE_BOX_RIGHT" and self._box_rotation_ready():
            self.phase = "ALIGN_BEHIND_ROTATED_BOX"
            self.step = 0
        elif self.phase == "ROTATE_BOX_RIGHT" and self.step >= self.ROTATE_PUSH_PULSE_STEPS:
            self.rotate_pulse_count += 1
            self._update_rotate_strategy_after_pulse()
            self.phase = "ROTATE_RELEASE_OBSERVE"
            self.step = 0
        elif self.phase == "ROTATE_RELEASE_OBSERVE" and self.step >= self.ROTATE_RELEASE_OBSERVE_STEPS:
            if self._box_rotation_ready():
                self.phase = "ALIGN_BEHIND_ROTATED_BOX"
            elif self.rotate_pulse_count >= self.MAX_ROTATE_PULSES:
                self.phase = "MOVE_FORWARD_BESIDE_BOX"
                self.rotate_pulse_count = 0
                self._advance_rotate_strategy()
            else:
                self.phase = "ROTATE_BOX_RIGHT"
            self.rotate_no_progress_ticks = 0
            self.rotate_release_ticks = 0
            self.step = 0
        elif self.phase == "ALIGN_BEHIND_ROTATED_BOX" and (
            (self._behind_rotated_box_ready() and self._box_rotation_ready())
            or self.step >= self.ALIGN_BEHIND_BOX_MAX_STEPS
        ):
            self.phase = "INSERT_BOX_TO_HOLE" if self._box_rotation_ready() else "ROTATE_BOX_RIGHT"
            self.rotate_no_progress_ticks = 0
            self.rotate_release_ticks = 0
            self.step = 0
        elif self.phase == "INSERT_BOX_TO_HOLE" and not self._box_rotation_ready():
            self.phase = "ROTATE_BOX_RIGHT"
            self.rotate_no_progress_ticks = 0
            self.rotate_release_ticks = 0
            self.box_yaw_confidence = min(self.box_yaw_confidence, 0.20)
            self.step = 0
        elif self.phase == "INSERT_BOX_TO_HOLE" and (
            self._box_inserted_enough() or self.step >= self.INSERT_BOX_MAX_STEPS
        ):
            self.bridge_ready = self._bridge_pose_ready()
            self.insert_attempts += 1
            self.rotation_session_active = False
            self.phase = "RELEASE_BOX"
            self.step = 0
        elif self.phase == "RELEASE_BOX" and (
            (self.step >= self.POST_INSERT_BACKUP_STEPS and self.est_x <= self.RELEASE_SAFE_X)
            or self.step >= self.POST_INSERT_BACKUP_MAX_STEPS
        ):
            self.bridge_ready = self._bridge_pose_ready()
            if self.bridge_ready:
                self.phase = "CROSS"
            else:
                if self.insert_attempts >= self.MAX_INSERT_ATTEMPTS:
                    self.insert_attempts = 0
                self.phase = "ALIGN_BEHIND_ROTATED_BOX"
            self.step = 0

        if self.phase == "BACK_UP":
            action = self._back_up_action(obs, action_dim)
        elif self.phase == "MOVE_LEFT_TO_BOX_LANE":
            action = self._move_left_to_box_lane_action(obs, action_dim)
        elif self.phase == "CONTACT_BOX":
            action = self._contact_box_action(obs, action_dim)
        elif self.phase == "PUSH_BOX":
            action = self._push_box_action(obs, action_dim)
        elif self.phase == "DETACH_FROM_BOX":
            action = self._detach_from_box_action(obs, action_dim)
        elif self.phase == "MOVE_LEFT_OF_BOX":
            action = self._move_left_of_box_action(obs, action_dim)
        elif self.phase == "CENTER_BOX_Y":
            action = self._center_box_y_action(obs, action_dim)
        elif self.phase == "MOVE_FORWARD_BESIDE_BOX":
            action = self._move_forward_beside_box_action(obs, action_dim)
        elif self.phase == "ROTATE_BOX_RIGHT":
            action = self._rotate_box_right_action(obs, action_dim)
        elif self.phase == "ROTATE_RELEASE_OBSERVE":
            action = self._rotate_release_observe_action(obs, action_dim)
        elif self.phase == "RETREAT_FROM_PIT":
            action = self._retreat_from_pit_action(obs, action_dim)
        elif self.phase == "ALIGN_BEHIND_ROTATED_BOX":
            action = self._align_behind_rotated_box_action(obs, action_dim)
        elif self.phase == "INSERT_BOX_TO_HOLE":
            action = self._insert_box_to_hole_action(obs, action_dim)
        elif self.phase == "RELEASE_BOX":
            action = self._release_box_action(obs, action_dim)
        else:
            action = self._cross_action(obs, action_dim)

        self._log_phase_command()
        self.step += 1 
        return {"action": action.cpu().tolist(), "giveup": False}
    
    def _back_up_action(self, obs, action_dim: int) -> torch.Tensor:
        """Create room behind the box before moving sideways into the box lane."""
        self._set_velocity_command(-1.0, 0.0, 0.0)
        return self._compute_base_action(obs, action_dim)

    def _move_left_to_box_lane_action(self, obs, action_dim: int) -> torch.Tensor:
        """Move left, then use sensors to center the visible box."""
        lin_y = self._depth_corrected_lateral_cmd(obs, base_lin_y=0.45, gain=0.45)
        lin_y = self._lidar_corrected_lateral_cmd(lin_y, gain=0.18)
        self._set_velocity_command(-0.18, lin_y, 0.0)
        return self._compute_base_action(obs, action_dim)

    def _contact_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Creep forward only enough to touch/locate the box before side setup."""
        lin_y = self._depth_corrected_lateral_cmd(obs, base_lin_y=0.0, gain=0.30)
        lin_y = self._lidar_corrected_lateral_cmd(lin_y, gain=0.12)
        yaw_cmd = float(max(-0.30, min(0.30, -1.2 * self.est_yaw)))
        self._set_velocity_command(0.35, lin_y, yaw_cmd)
        return self._compute_base_action(obs, action_dim)
    
    def _push_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Use a stronger +X command to push the box into the scoring x-range."""
        lin_y = self._depth_corrected_lateral_cmd(obs, base_lin_y=0.0, gain=0.25)
        lin_y = self._lidar_corrected_lateral_cmd(lin_y, gain=0.10)
        yaw_cmd = float(max(-0.30, min(0.30, -1.4 * self.est_yaw)))
        self._set_velocity_command(1.00, lin_y, yaw_cmd)
        base_action = self._compute_base_action(obs, action_dim)
        return torch.clamp(base_action, -1.0, 1.0)

    def _detach_from_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Back away from the box while correcting yaw to face straight."""
        yaw_cmd = float(max(-0.35, min(0.35, -1.5 * self.est_yaw)))
        self._set_velocity_command(-0.60, 0.0, yaw_cmd)
        return self._compute_base_action(obs, action_dim)

    def _move_left_of_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Move to the +Y side of the box after contact has been released."""
        yaw_cmd = float(max(-0.30, min(0.30, -1.2 * self.est_yaw)))
        self._set_velocity_command(-0.05, 0.75, yaw_cmd)
        return self._compute_base_action(obs, action_dim)

    def _center_box_y_action(self, obs, action_dim: int) -> torch.Tensor:
        """Push the box toward the hole's Y center before rotating it."""
        yaw_cmd = float(max(-0.35, min(0.35, -1.5 * self.est_yaw)))
        y_error = self.box_est_y - self.BOX_INSERT_TARGET_Y

        # Only recover upward if we are clearly no longer on the +Y side. Avoid
        # oscillating left-right while the box needs one clean side push.
        if self.est_y < self.BOX_LEFT_SIDE_Y - 0.35:
            self._set_velocity_command(-0.05, 0.55, yaw_cmd)
            return self._compute_base_action(obs, action_dim)

        # Positive y_error means the box is left/high of the hole center, so the
        # robot pushes it right/down in -Y while keeping slight forward contact.
        if y_error > self.CENTER_BOX_Y_TOL:
            lin_y = -0.95
        elif y_error < -self.CENTER_BOX_Y_TOL:
            lin_y = 0.10
        else:
            lin_y = -0.30
        self._set_velocity_command(0.16, lin_y, yaw_cmd)
        base_action = self._compute_base_action(obs, action_dim)
        return torch.clamp(base_action, -1.0, 1.0)

    def _move_forward_beside_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Move forward on the left side until reaching an off-center rotate point."""
        y_error = self.BOX_LEFT_SIDE_Y - self.est_y
        x_error = self._rotate_x_target() - self.est_x
        lin_y = float(max(-0.35, min(0.35, 0.9 * y_error)))
        if self.est_y > self.BOX_LEFT_SIDE_Y + 0.18:
            lin_y = min(lin_y, -0.35)
        lin_x = float(max(-0.25, min(0.90, 0.85 * x_error)))
        if abs(x_error) < 0.10:
            lin_x = 0.0
        yaw_cmd = float(max(-0.30, min(0.30, -1.2 * self.est_yaw)))
        self._set_velocity_command(lin_x, lin_y, yaw_cmd)
        return self._compute_base_action(obs, action_dim)

    def _rotate_box_right_action(self, obs, action_dim: int) -> torch.Tensor:
        """Short push pulse using the active rotate primitive."""
        yaw_cmd = float(max(-0.35, min(0.35, -1.6 * self.est_yaw)))
        strategy = self._rotate_strategy_name()
        if (
            strategy in ("LEFT_CORNER_CW", "SIDE_SWEEP")
            and not self._at_rotate_lane()
            and self.step == 0
        ):
            self._set_velocity_command(-0.25, 0.75, yaw_cmd)
            return self._compute_base_action(obs, action_dim)

        if not self.bridge_ready and self.est_x > self.PIT_GUARD_X:
            self._set_velocity_command(-0.45, 0.15, yaw_cmd)
            return self._compute_base_action(obs, action_dim)

        yaw_error = abs(self.BOX_ROTATE_TARGET_YAW - self.box_est_yaw)
        forward_cmd = float(max(0.40, min(0.62, 0.42 + 0.12 * yaw_error)))
        if strategy == "LEFT_CORNER_CW":
            side_cmd = -1.0
            if self.lidar_box is not None:
                side_cmd = float(max(-1.0, min(-0.55, -0.80 - 0.15 * self.lidar_box["bearing"])))
        elif strategy == "RIGHT_CORNER_CCW":
            # Alternate primitive: approach from the lower-Y side and try the
            # opposite corner if the left-corner contact is not producing yaw.
            target_y = self.BOX_INSERT_TARGET_Y - 0.35
            if self.est_y > target_y + 0.20:
                self._set_velocity_command(-0.20, -0.75, yaw_cmd)
                return self._compute_base_action(obs, action_dim)
            forward_cmd = float(max(0.30, min(0.48, 0.34 + 0.08 * yaw_error)))
            side_cmd = 0.75
        else:
            # Side sweep: shallow push across the face to move contact toward a
            # corner before going back to a corner primitive.
            forward_cmd = 0.28
            side_cmd = -0.55 if self.est_y >= self.BOX_INSERT_TARGET_Y else 0.55
        self._set_velocity_command(forward_cmd, side_cmd, yaw_cmd)
        base_action = self._compute_base_action(obs, action_dim)
        return torch.clamp(base_action, -1.0, 1.0)

    def _rotate_release_observe_action(self, obs, action_dim: int) -> torch.Tensor:
        """Release contact after a rotate pulse so LiDAR can re-observe the box."""
        yaw_cmd = float(max(-0.35, min(0.35, -1.5 * self.est_yaw)))
        strategy = self._rotate_strategy_name()
        target_y = self.BOX_INSERT_TARGET_Y - 0.35 if strategy == "RIGHT_CORNER_CCW" else min(self.BOX_LEFT_SIDE_Y, self.BOX_EST_Y_MAX)
        y_error = target_y - self.est_y
        lin_y = float(max(-0.35, min(0.25, 0.7 * y_error)))
        self._set_velocity_command(-0.38, lin_y, yaw_cmd)
        return self._compute_base_action(obs, action_dim)

    def _retreat_from_pit_action(self, obs, action_dim: int) -> torch.Tensor:
        """Move back to a safe x before attempting more box manipulation."""
        yaw_cmd = float(max(-0.35, min(0.35, -1.5 * self.est_yaw)))
        target_y = min(self.BOX_LEFT_SIDE_Y, self.BOX_EST_Y_MAX)
        y_error = target_y - self.est_y
        lin_y = float(max(-0.45, min(0.25, 0.8 * y_error)))
        self._set_velocity_command(-0.65, lin_y, yaw_cmd)
        return self._compute_base_action(obs, action_dim)

    def _insert_box_to_hole_action(self, obs, action_dim: int) -> torch.Tensor:
        """Push from behind the rotated box toward the pit/hole lane."""
        yaw_cmd = float(max(-0.35, min(0.35, -1.6 * self.est_yaw)))
        y_error = self.box_est_y - self.BOX_INSERT_TARGET_Y
        # At this point the robot should be behind the rotated box, so insertion
        # should mostly be a straight -Y push, not another corner rotation.
        forward_cmd = 0.00
        side_cmd = float(max(-0.95, min(-0.25, -0.55 - 0.60 * y_error)))
        if self.box_est_y < self.BOX_INSERT_TARGET_Y - self.BOX_INSERT_Y_TOL:
            # Box is already too far right/low in Y; stop forcing it into the wall.
            side_cmd = max(side_cmd, -0.20)
        elif self.box_est_y > self.BOX_INSERT_TARGET_Y + self.BOX_INSERT_Y_TOL:
            # Box is too far left/high in Y; push harder toward the hole center.
            side_cmd = min(side_cmd, -0.75)
        if self.lidar_box is not None:
            # Keep the box roughly centered while pushing from behind.
            forward_cmd = float(max(-0.12, min(0.12, 0.08 * self.lidar_box["bearing"])))
        if self.est_x > self.INSERT_MAX_ROBOT_X:
            forward_cmd = -0.20
        self._set_velocity_command(forward_cmd, side_cmd, yaw_cmd)
        base_action = self._compute_base_action(obs, action_dim)
        return torch.clamp(base_action, -1.0, 1.0)

    def _behind_rotated_box_ready(self) -> bool:
        """Check whether robot has left the corner and is behind the rotated box."""
        if self.step < self.ALIGN_BEHIND_BOX_MIN_STEPS:
            return False
        safe_x = self.est_x <= self.INSERT_MAX_ROBOT_X + 0.05
        behind_y = self.est_y >= min(self.BOX_EST_Y_MAX, self.box_est_y + 0.45)
        lidar_ok = self.lidar_box is None or abs(self.lidar_box["bearing"]) < 1.45
        return safe_x and behind_y and lidar_ok

    def _align_behind_rotated_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Move off the corner contact and line up behind the rotated box."""
        target_x = min(self.INSERT_MAX_ROBOT_X - 0.05, self.box_est_x - 0.05)
        target_y = min(self.BOX_EST_Y_MAX, self.box_est_y + 0.45)
        x_error = target_x - self.est_x
        y_error = target_y - self.est_y
        lin_x = float(max(-0.45, min(0.25, 0.9 * x_error)))
        lin_y = float(max(-0.35, min(0.35, 0.75 * y_error)))
        if self.est_y > self.BOX_EST_Y_MAX:
            lin_y = min(lin_y, -0.20)
        yaw_cmd = float(max(-0.35, min(0.35, -1.5 * self.est_yaw)))
        self._set_velocity_command(lin_x, lin_y, yaw_cmd)
        return self._compute_base_action(obs, action_dim)

    def _box_seen_on_right_side(self) -> bool:
        """Return True once LiDAR suggests the box has moved to robot's right side."""
        if self.lidar_box is None:
            return False
        return self.lidar_box["bearing"] < -0.35 or self.lidar_bearing_delta < -0.015

    def _box_is_not_centered_front(self) -> bool:
        """Insertion likely progressed when the box is no longer centered ahead."""
        if self.lidar_box is None:
            return False
        return abs(self.lidar_box["bearing"]) > 0.55

    def _release_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Back away after insertion so the robot does not stay wedged on the box."""
        yaw_cmd = float(max(-0.35, min(0.35, -1.5 * self.est_yaw)))
        self._set_velocity_command(-0.45, 0.30, yaw_cmd)
        return self._compute_base_action(obs, action_dim)

    def _cross_action(self, obs, action_dim: int) -> torch.Tensor:
        """Continue forward after the box interaction attempt."""
        if not self.bridge_ready:
            # Last-resort crossing should be cautious; the global pit guard will
            # also clamp forward velocity if the bridge is not confirmed.
            self._set_velocity_command(0.20, 0.0, 0.0)
        else:
            self._set_velocity_command(0.75, 0.0, 0.0)
        return self._compute_base_action(obs, action_dim)

        # with torch.inference_mode():
        #     action_train = self.policy(policy_obs)

        # if not isinstance(action_train, torch.Tensor):
        #     action_train = torch.as_tensor(
        #         action_train, device=self.device, dtype=torch.float32
        #     )

        # action_train = action_train.to(device=self.device, dtype=torch.float32)

        # if action_train.ndim == 1:
        #     action_train = action_train.unsqueeze(0)

        # action_env = self._map_policy_action_to_env_action(action_train, action_dim)
        # action_env = action_env.cpu().numpy().tolist()
        # return {'action': action_env, 'giveup': False}
