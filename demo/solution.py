import os
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

        self.BACK_UP_STEPS = 110
        self.MOVE_LEFT_STEPS = 320
        self.CONTACT_STEPS = 55
        self.PUSH_BOX_STEPS = 360
        self.depth_box = None


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
        self.fixed_velocity_commands = torch.tensor(
            [lin_x, lin_y, ang_z], device=self.device, dtype=torch.float32
        ).view(1, 3)

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
        depth = self._get_image_tensor(obs, "head_depth", "video_depth", "ee_depth")
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
        """Estimate horizontal image error of the nearest box-like object in head depth."""
        depth = self._get_depth_image(obs)
        if depth is None:
            self.depth_box = None
            return None

        height, width = depth.shape
        row0, row1 = int(height * 0.30), int(height * 0.88)
        col0, col1 = int(width * 0.08), int(width * 0.92)
        roi = depth[row0:row1, col0:col1]

        valid = torch.isfinite(roi) & (roi > 0.15) & (roi < 5.0)
        valid_depth = roi[valid]
        if valid_depth.numel() < 200:
            self.depth_box = None
            return None

        # The box should be one of the nearest large objects in the head camera.
        flat = valid_depth.flatten()
        kth = max(1, int(flat.numel() * 0.18))
        near_depth = torch.kthvalue(flat, kth).values + 0.25
        near_mask = valid & (roi <= near_depth)
        if near_mask.sum().item() < 120:
            self.depth_box = None
            return None

        _ys, xs = torch.where(near_mask)
        xs = xs.to(torch.float32) + float(col0)
        center_x = xs.mean()
        image_center_x = torch.tensor(float(width - 1) * 0.5, device=self.device)
        x_error = ((center_x - image_center_x) / image_center_x).clamp(-1.0, 1.0)
        distance = roi[near_mask].median()

        estimate = {
            "x_error": float(x_error.item()),
            "distance": float(distance.item()),
            "pixels": int(near_mask.sum().item()),
        }
        self.depth_box = estimate
        return estimate

    def _box_centered_from_depth(self, obs) -> bool:
        estimate = self._estimate_box_from_depth(obs)
        if estimate is None:
            return False
        return abs(estimate["x_error"]) < 0.16 and estimate["distance"] < 3.0

    def _depth_corrected_lateral_cmd(self, obs, base_lin_y: float, gain: float = 0.35) -> float:
        estimate = self._estimate_box_from_depth(obs)
        if estimate is None:
            return base_lin_y
        # Negative image error means the box appears left; command +Y to move left.
        corrected = base_lin_y - gain * estimate["x_error"]
        return float(max(-0.55, min(0.55, corrected)))

    def predicts(self, obs, current_score):
        """Run policy inference and return current-env full-body action."""
        # if current_score > 1:
        #     return {'action': [], 'giveup': True}
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        if self.phase == "BACK_UP":
            action = self._back_up_action(obs, action_dim)
            if self.step >= self.BACK_UP_STEPS:
                self.phase = "MOVE_LEFT_TO_BOX_LANE"
                self.step = 0
        elif self.phase == "MOVE_LEFT_TO_BOX_LANE":
            action = self._move_left_to_box_lane_action(obs, action_dim)
            if self._box_centered_from_depth(obs) or self.step >= self.MOVE_LEFT_STEPS:
                self.phase = "CONTACT_BOX"
                self.step = 0
        elif self.phase == "CONTACT_BOX":
            action = self._contact_box_action(obs, action_dim)
            if self.step >= self.CONTACT_STEPS:
                self.phase = "PUSH_BOX"
                self.step = 0 
        elif self.phase == "PUSH_BOX":
            action = self._push_box_action(obs, action_dim)
            if current_score >= 16.0 or self.step >= self.PUSH_BOX_STEPS:
                self.phase = "CROSS"
                self.step = 0
        else:
            action = self._cross_action(obs, action_dim)

        self.step += 1 
        return {"action": action.cpu().tolist(), "giveup": False}
    
    def _back_up_action(self, obs, action_dim: int) -> torch.Tensor:
        """Create room behind the box before moving sideways into the box lane."""
        self._set_velocity_command(-0.35, 0.0, 0.0)
        return self._compute_base_action(obs, action_dim)

    def _move_left_to_box_lane_action(self, obs, action_dim: int) -> torch.Tensor:
        """Move left, then use depth to center the visible box in the head camera."""
        lin_y = self._depth_corrected_lateral_cmd(obs, base_lin_y=0.45, gain=0.45)
        self._set_velocity_command(0.0, lin_y, 0.0)
        return self._compute_base_action(obs, action_dim)

    def _contact_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Creep forward to make contact before the strong push phase."""
        lin_y = self._depth_corrected_lateral_cmd(obs, base_lin_y=0.0, gain=0.30)
        self._set_velocity_command(0.25, lin_y, 0.0)
        return self._compute_base_action(obs, action_dim)
    
    def _push_box_action(self, obs, action_dim: int) -> torch.Tensor:
        """Use a stronger +X command to push the box into the scoring x-range."""
        lin_y = self._depth_corrected_lateral_cmd(obs, base_lin_y=0.0, gain=0.25)
        self._set_velocity_command(0.85, lin_y, 0.0)
        base_action = self._compute_base_action(obs, action_dim)
        return torch.clamp(base_action, -1.0, 1.0)

    def _cross_action(self, obs, action_dim: int) -> torch.Tensor:
        """Continue forward after the box interaction attempt."""
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
