"""
Task D solution: Box rotation for pit crossing.

Built on top of solution_rl.py (the working baseline).
The RL policy works when velocity commands are injected properly as a 2D tensor.
We add a state machine to change the velocity command per phase.
"""

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

        self.arm_default_action = torch.zeros(
            (1, self.arm_action_dim),
            device=self.device,
            dtype=torch.float32,
        )

        # ── Velocity command (can be changed per phase) ───────────────────────
        self._vel_x = 0.5
        self._vel_y = 0.0
        self._vel_z = 0.0

        # ── State machine ─────────────────────────────────────────────────────
        self.phase = "TO_BOX"
        self.step = 0
        self._stuck_ticks = 0

        # ── Phase tuning ───────────────────────────────────────────────────────
        # Step counts for each phase (tuned so robot moves real distances)
        self.STEPS_TO_BOX = 200    # walk diagonally toward box's Y
        self.STEPS_CONTACT = 100   # walk forward into box
        self.STEPS_PUSH = 300     # push box toward pit
        self.STEPS_DETACH = 80    # back away
        self.STEPS_TO_SIDE = 200   # move to +Y side of box
        self.STEPS_ROTATE_PUSH = 80
        self.STEPS_ROTATE_RELEASE = 50
        self.MAX_ROTATE_PULSES = 10

        # ── Rotation sub-state ─────────────────────────────────────────────────
        self._rotate_pulses = 0
        self._rotate_substep = "push"

        # ── Diagnostic ────────────────────────────────────────────────────────
        self._last_phase = None
        self._printed_obs = False

    # --------------------------------------------------------------
    # Velocity command (mirrors solution_rl.py pattern exactly)
    # --------------------------------------------------------------

    def _get_velocity_commands(self, proprio: torch.Tensor) -> torch.Tensor:
        """Return the current velocity command as a 2D tensor (num_envs, 3)."""
        num_envs = int(proprio.shape[0])

        cmd = torch.tensor(
            [self._vel_x, self._vel_y, self._vel_z],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)

        if num_envs > 1:
            cmd = cmd.repeat(num_envs, 1)
        return cmd

    # --------------------------------------------------------------
    # Policy observation (exactly matches solution_rl.py)
    # --------------------------------------------------------------

    def _extract_policy_obs(self, obs, action_dim) -> torch.Tensor:
        """Build the 45-dim policy observation vector.

        Exactly mirrors solution_rl.py — proprietary stays 2D,
        velocity commands injected as 2D (num_envs, 3).
        """
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

    # --------------------------------------------------------------
    # Map policy output to env action (exactly matches solution_rl.py)
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Phase actions — set velocity command, run policy
    # --------------------------------------------------------------

    def _phase_action(self, obs, action_dim: int, lin_x: float, lin_y: float, ang_z: float) -> torch.Tensor:
        """Set velocity command and run the RL policy."""
        self._vel_x = lin_x
        self._vel_y = lin_y
        self._vel_z = ang_z

        policy_obs = self._extract_policy_obs(obs, action_dim)
        with torch.inference_mode():
            action_train = self.policy(policy_obs)

        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        return self._map_policy_action_to_env_action(action_train, action_dim)

    # --------------------------------------------------------------
    # State machine transitions
    # --------------------------------------------------------------

    def _transition(self) -> None:
        p = self.phase
        s = self.step

        if p == "TO_BOX":
            if s >= self.STEPS_TO_BOX:
                self.phase = "CONTACT"
                self.step = 0

        elif p == "CONTACT":
            if s >= self.STEPS_CONTACT:
                self.phase = "PUSH_FWD"
                self.step = 0

        elif p == "PUSH_FWD":
            if s >= self.STEPS_PUSH:
                self.phase = "DETACH"
                self.step = 0

        elif p == "DETACH":
            if s >= self.STEPS_DETACH:
                self.phase = "TO_SIDE"
                self.step = 0

        elif p == "TO_SIDE":
            if s >= self.STEPS_TO_SIDE:
                self.phase = "ROTATE"
                self.step = 0
                self._rotate_pulses = 0
                self._rotate_substep = "push"

        elif p == "ROTATE":
            if self._rotate_substep == "push":
                if s >= self.STEPS_ROTATE_PUSH:
                    self._rotate_substep = "release"
                    self.step = 0
            elif self._rotate_substep == "release":
                if s >= self.STEPS_ROTATE_RELEASE:
                    self._rotate_pulses += 1
                    if self._rotate_pulses >= self.MAX_ROTATE_PULSES:
                        self.phase = "CROSS"
                        self.step = 0
                    else:
                        self._rotate_substep = "push"
                        self.step = 0

        elif p == "CROSS":
            pass

    # --------------------------------------------------------------
    # Main entry point
    # --------------------------------------------------------------

    def predicts(self, obs, current_score):
        if not self._printed_obs:
            print("OBS KEYS:", list(obs.keys()))
            self._printed_obs = True

        if current_score > 1:
            return {'action': [], 'giveup': True}

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        self._transition()

        p = self.phase

        if p == "TO_BOX":
            # Forward + left to reach box's Y position (box at y=1.6)
            action = self._phase_action(obs, action_dim, lin_x=0.5, lin_y=0.6, ang_z=0.0)
        elif p == "CONTACT":
            # Pure forward to push the box
            action = self._phase_action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)
        elif p == "PUSH_FWD":
            action = self._phase_action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)
        elif p == "DETACH":
            action = self._phase_action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)
        elif p == "TO_SIDE":
            action = self._phase_action(obs, action_dim, lin_x=0.5, lin_y=0.5, ang_z=0.0)
        elif p == "ROTATE":
            action = self._phase_action(obs, action_dim, lin_x=0.5, lin_y=-0.3, ang_z=0.0)
        elif p == "CROSS":
            action = self._phase_action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)
        else:
            action = self._phase_action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)

        # Log on phase change or every 100 steps
        if p != self._last_phase:
            print(f"[D] phase={p:<12} step={self.step}  cmd=({self._vel_x}, {self._vel_y}, {self._vel_z})")
            self._last_phase = p

        self.step += 1
        return {"action": action.cpu().numpy().tolist(), "giveup": False}
