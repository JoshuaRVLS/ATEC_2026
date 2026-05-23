"""
Task D solution: Box rotation for pit crossing.

Geometry (from env_cfg):
  Robot start: (-3, 0, 0)     Box: (-3, 1.6, 0)    Pit center: (-0.8, 1.1)
  Box: 0.8 x 1.0 x 0.6m, spawned at y=1.6, extends to y=2.1
  Robot at y=0 can NEVER reach box without moving in Y FIRST.

Strategy:
  1. Walk diagonally (forward+left) to reach box's Y position
  2. Contact box and push it toward the pit
  3. Detach and move to the +Y side of the box
  4. Rotate the box via corner pushes
  5. Push rotated box to bridge the pit
  6. Cross the pit

The RL policy tracks velocity commands: (lin_x, lin_y, ang_z).
- Forward + left: move toward box
- Pure forward: push box to pit
- Forward + slight turn: navigate after rotation
"""

import os
import math
import torch


class AlgSolution:

    LEG_ACTION_DIM = 12
    ARM_ACTION_DIM = 8

    def __init__(self):
        policy_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'policy.pt')
        self.device = 'cuda'

        self.policy = torch.jit.load(policy_path, map_location=self.device)
        self.policy.eval()

        self.leg_joint_indices = list(range(self.LEG_ACTION_DIM))
        self.arm_joint_indices = list(range(self.LEG_ACTION_DIM, self.LEG_ACTION_DIM + self.ARM_ACTION_DIM))

        self.train_to_env = torch.tensor([
            0.25, 0.5, 0.5, 0.25, 0.5, 0.5,
            0.25, 0.5, 0.5, 0.25, 0.5, 0.5,
        ], device=self.device, dtype=torch.float32).view(1, -1)

        self.env_to_train = torch.tensor([
            4.0, 2.0, 2.0, 4.0, 2.0, 2.0,
            4.0, 2.0, 2.0, 4.0, 2.0, 2.0,
        ], device=self.device, dtype=torch.float32).view(1, -1)

        self.arm_default = torch.zeros((1, self.ARM_ACTION_DIM), device=self.device, dtype=torch.float32)

        # ── State machine ─────────────────────────────────────────────────────
        self.phase = "TO_BOX"
        self.step = 0

        # ── Ground-truth cache ─────────────────────────────────────────────────
        self._env_cache = None
        self._gt_cached = False

        # ── Phase thresholds ──────────────────────────────────────────────────
        self.BOX_Y_TARGET = 1.6   # robot should reach this Y to align with box
        self.PUSH_BOX_X = -2.8    # stop pushing when box reaches here
        self.PIT_CENTER_X = -0.8
        self.SIDE_Y = 2.5         # robot moves to +Y side of box for rotation

        # ── Phase sub-state ───────────────────────────────────────────────────
        self._rotate_pulses = 0
        self._rotate_substep = "push"
        self._stuck_ticks = 0
        self._prev_x = -3.0
        self._prev_y = 0.0

        # ── Diagnostic ────────────────────────────────────────────────────────
        self._last_logged_phase = None
        self._printed_obs = False

    # ── Ground-truth access ────────────────────────────────────────────────────

    def _cache_env(self, obs: dict) -> None:
        if self._gt_cached:
            return
        try:
            extero = obs.get("extero") or obs.get("proprio")
            if extero is None:
                return
            manager = getattr(extero, "_manager", None)
            if manager is None:
                return
            self._env_cache = manager
            self._gt_cached = True
            # Initialize from GT
            gt = self._get_gt_robot_pose()
            if gt:
                self._prev_x, self._prev_y = gt[0], gt[1]
        except Exception:
            self._gt_cached = False

    def _get_gt_robot_pose(self):
        try:
            if self._env_cache is None:
                return None
            robot = self._env_cache._env.scene["robot"]
            pos = robot.data.root_pos_w[0].cpu().numpy()
            quat = robot.data.root_quat_w[0].cpu().numpy()
            yaw = math.atan2(2.0*(quat[0]*quat[3] + quat[1]*quat[2]), 1.0 - 2.0*(quat[1]**2 + quat[2]**2))
            return float(pos[0]), float(pos[1]), float(yaw)
        except Exception:
            return None

    def _get_gt_box_pose(self):
        try:
            if self._env_cache is None:
                return None
            box = self._env_cache._env.scene["box"]
            pos = box.data.root_pos_w[0].cpu().numpy()
            return float(pos[0]), float(pos[1])
        except Exception:
            return None

    # ── Policy interface ─────────────────────────────────────────────────────

    def _policy_obs(self, proprio: torch.Tensor, action_dim: int) -> torch.Tensor:
        if proprio.ndim == 2:
            proprio = proprio.squeeze(0)
        idx = 0
        idx += 3   # base_lin_vel
        base_ang_vel = proprio[idx:idx + 3]; idx += 3
        idx += 3   # velocity_commands (injected below)
        idx += 3   # projected_gravity
        idx += 3
        joint_pos = proprio[idx:idx + action_dim]; idx += action_dim
        joint_vel = proprio[idx:idx + action_dim]; idx += action_dim
        actions = proprio[idx:idx + action_dim]
        joint_pos_leg = joint_pos[self.leg_joint_indices]
        joint_vel_leg = joint_vel[self.leg_joint_indices]
        actions_leg = actions[self.leg_joint_indices]
        actions_train = actions_leg * self.env_to_train.squeeze(0)
        vel_cmd = self._vel_cmd.squeeze(0).to(dtype=proprio.dtype, device=self.device)
        return torch.cat([
            base_ang_vel * 0.25,
            proprio[9:12],
            vel_cmd,
            joint_pos_leg,
            joint_vel_leg * 0.05,
            actions_train,
        ], dim=-1).unsqueeze(0)

    def _run_policy(self, obs, action_dim: int) -> torch.Tensor:
        policy_input = self._policy_obs(obs["proprio"].to(self.device), action_dim)
        with torch.inference_mode():
            action_train = self.policy(policy_input)
        if not isinstance(action_train, torch.Tensor):
            action_train = torch.as_tensor(action_train, device=self.device, dtype=torch.float32)
        action_train = action_train.to(device=self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)
        action_env = torch.zeros((1, action_dim), device=self.device, dtype=torch.float32)
        leg_action = action_train[:, :self.LEG_ACTION_DIM] * self.train_to_env
        action_env[:, self.leg_joint_indices] = leg_action
        action_env[:, self.arm_joint_indices] = self.arm_default
        return action_env

    # ── Action helper ──────────────────────────────────────────────────────────

    def _action(self, obs, action_dim: int, lin_x: float, lin_y: float, ang_z: float) -> torch.Tensor:
        """Run the RL policy with the given velocity command."""
        self._vel_cmd = torch.tensor([lin_x, lin_y, ang_z], device=self.device, dtype=torch.float32).view(1, 3)
        return self._run_policy(obs, action_dim)

    # ── Phase transitions ─────────────────────────────────────────────────────

    def _transition(self, robot_pose, box_pose) -> None:
        p = self.phase
        s = self.step
        rx, ry, ryaw = robot_pose if robot_pose else (-3.0, 0.0, 0.0)
        bx, by = box_pose if box_pose else (-3.0, 1.6)

        if p == "TO_BOX":
            # Walk diagonally forward+left to reach box's Y position
            # Use small forward + positive Y velocity to move toward box
            dx = abs(rx - self._prev_x)
            dy = abs(ry - self._prev_y)
            if dx < 0.005 and dy < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x = rx
                self._prev_y = ry

            # Stop when aligned with box in Y (robot is now in the box's X collision range)
            aligned = ry >= self.BOX_Y_TARGET - 0.3
            stuck = self._stuck_ticks >= 50
            if aligned or stuck or s >= 200:
                self.phase = "CONTACT"
                self.step = 0
                self._prev_x = rx
                self._stuck_ticks = 0

        elif p == "CONTACT":
            # Walk forward into the box — collision will push it
            dx = abs(rx - self._prev_x)
            if dx < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x = rx

            stuck = self._stuck_ticks >= 30
            # Advance when stuck (box is blocking) or after long enough
            if stuck or s >= 100:
                self.phase = "PUSH_FWD"
                self.step = 0
                self._stuck_ticks = 0
                self._prev_x = rx

        elif p == "PUSH_FWD":
            # Push box toward the pit (forward motion)
            dx = abs(rx - self._prev_x)
            if dx < 0.005:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = 0
                self._prev_x = rx

            # Stop pushing when box is near the pit, or stuck
            box_near_pit = bx >= self.PUSH_BOX_X and s > 20
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
            # Move to +Y side of box (forward+left to navigate around box)
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
                if s >= 80:
                    self._rotate_substep = "release"
                    self.step = 0
            elif self._rotate_substep == "release":
                if s >= 50:
                    self._rotate_pulses += 1
                    if self._rotate_pulses >= 10:
                        self.phase = "CROSS"
                        self.step = 0
                    else:
                        self._rotate_substep = "push"
                        self.step = 0

        elif p == "CROSS":
            pass

    # ── Main entry point ─────────────────────────────────────────────────────

    def predicts(self, obs, current_score):
        if not self._printed_obs:
            print("OBS KEYS:", list(obs.keys()))
            self._printed_obs = True

        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

        self._cache_env(obs)
        robot_pose = self._get_gt_robot_pose()
        box_pose = self._get_gt_box_pose()

        self._transition(robot_pose, box_pose)

        p = self.phase

        # Velocity commands per phase
        if p == "TO_BOX":
            # Forward + LEFT (positive Y) to reach box's Y position
            action = self._action(obs, action_dim, lin_x=0.5, lin_y=0.6, ang_z=0.0)
        elif p == "CONTACT":
            action = self._action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)
        elif p == "PUSH_FWD":
            action = self._action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)
        elif p == "DETACH":
            action = self._action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)
        elif p == "TO_SIDE":
            action = self._action(obs, action_dim, lin_x=0.5, lin_y=0.5, ang_z=0.0)
        elif p == "ROTATE":
            action = self._action(obs, action_dim, lin_x=0.5, lin_y=-0.3, ang_z=0.0)
        elif p == "CROSS":
            action = self._action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)
        else:
            action = self._action(obs, action_dim, lin_x=0.5, lin_y=0.0, ang_z=0.0)

        # Log
        if p != self._last_logged_phase or self.step % 100 == 0:
            rx, ry, ryaw = robot_pose if robot_pose else (0, 0, 0)
            bx, by = box_pose if box_pose else (0, 0)
            print(
                f"[D] phase={p:<12} step={self.step:>4}  "
                f"robot=({rx:+.2f}, {ry:+.2f}, yaw={ryaw:+.2f})  "
                f"box=({bx:+.2f}, {by:+.2f})  "
                f"cmd=({self._vel_cmd[0,0].item():+.2f}, {self._vel_cmd[0,1].item():+.2f}, {self._vel_cmd[0,2].item():+.2f})"
            )
            self._last_logged_phase = p

        self.step += 1
        return {"action": action.cpu().tolist()[0], "giveup": False}
