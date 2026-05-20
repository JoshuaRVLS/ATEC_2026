from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Train Task D high-level command policy with PPO.")
parser.add_argument("--task", type=str, default="ATEC-TaskD-B2Piper")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--iterations", type=int, default=500)
parser.add_argument("--horizon", type=int, default=128)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--value_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--save_every", type=int, default=25)
parser.add_argument("--out", type=str, default="demo/high_level_ppo.pt")
parser.add_argument("--ckpt_dir", type=str, default="logs/task_d_highlevel_ppo")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402


def quat_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(dim=-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def angle_error(a: torch.Tensor, b: float) -> torch.Tensor:
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


class LowLevelLocomotion:
    def __init__(self, device: str):
        self.device = device
        policy_path = REPO_ROOT / "demo" / "policy.pt"
        self.policy = torch.jit.load(str(policy_path), map_location=device)
        self.policy.eval()

        self.leg_action_dim = 12
        self.leg_joint_indices = list(range(12))
        self.arm_joint_indices = list(range(12, 20))
        self.train_to_env_action_scale = torch.tensor(
            [
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
                0.25, 0.5, 0.5,
            ],
            device=device,
            dtype=torch.float32,
        ).view(1, -1)
        self.env_to_train_action_scale = torch.tensor(
            [
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
                4.0, 2.0, 2.0,
            ],
            device=device,
            dtype=torch.float32,
        ).view(1, -1)

    def __call__(self, obs: dict, commands: torch.Tensor) -> torch.Tensor:
        proprio = obs["proprio"].to(self.device)
        action_dim = (int(proprio.shape[-1]) - 12) // 3

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

        policy_obs = torch.cat(
            [
                base_ang_vel * 0.25,
                projected_gravity,
                commands.to(dtype=proprio.dtype),
                joint_pos_leg,
                joint_vel_leg * 0.05,
                actions_train_leg,
            ],
            dim=-1,
        )

        with torch.inference_mode():
            action_train = self.policy(policy_obs).to(self.device, dtype=torch.float32)
        if action_train.ndim == 1:
            action_train = action_train.unsqueeze(0)

        leg_action_env = action_train * self.train_to_env_action_scale
        action_env = torch.zeros((proprio.shape[0], action_dim), device=self.device, dtype=torch.float32)
        action_env[:, self.leg_joint_indices] = leg_action_env
        if action_dim >= 20:
            action_env[:, self.arm_joint_indices] = 0.0
        return action_env


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.register_buffer("command_scale", torch.tensor([1.0, 1.0, 0.8], dtype=torch.float32))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def act(self, obs: torch.Tensor):
        dist = self.distribution(obs)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        command = torch.tanh(raw_action) * self.command_scale.to(obs.device)
        value = self.critic(obs).squeeze(-1)
        return raw_action, command, log_prob, value

    def evaluate(self, obs: torch.Tensor, raw_action: torch.Tensor):
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        command = torch.tanh(raw_action) * self.command_scale.to(obs.device)
        value = self.critic(obs).squeeze(-1)
        return command, log_prob, entropy, value

    def deterministic_command(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(obs)) * self.command_scale.to(obs.device)


def make_env():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return env


def get_scene_tensors(env, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    box = unwrapped.scene["box"]

    robot_pos = robot.data.root_pos_w[:, :3]
    robot_yaw = quat_yaw_wxyz(robot.data.root_quat_w).unsqueeze(-1)
    box_pos = box.data.root_pos_w[:, :3]
    box_yaw = quat_yaw_wxyz(box.data.root_quat_w).unsqueeze(-1)

    proprio = obs["proprio"].to(args_cli.device)
    base_lin_vel = proprio[:, 0:3]
    base_ang_vel = proprio[:, 3:6]
    projected_gravity = proprio[:, 9:12]
    rel_box = box_pos[:, :2] - robot_pos[:, :2]
    yaw_to_90 = torch.minimum(
        torch.abs(angle_error(box_yaw.squeeze(-1), math.pi / 2)),
        torch.abs(angle_error(box_yaw.squeeze(-1), -math.pi / 2)),
    ).unsqueeze(-1)

    policy_obs = torch.cat(
        [
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            robot_pos[:, :2],
            robot_yaw,
            box_pos[:, :2],
            box_yaw,
            rel_box,
            yaw_to_90,
        ],
        dim=-1,
    ).to(args_cli.device, dtype=torch.float32)

    return policy_obs, robot_pos, box_pos


def shaped_reward(
    env,
    env_reward: torch.Tensor,
    robot_pos: torch.Tensor,
    box_pos: torch.Tensor,
    prev_box_pos: torch.Tensor,
    box_yaw: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    yaw_err = torch.minimum(
        torch.abs(angle_error(box_yaw, math.pi / 2)),
        torch.abs(angle_error(box_yaw, -math.pi / 2)),
    )
    prev_x = prev_box_pos[:, 0]
    box_x = box_pos[:, 0]
    box_y = box_pos[:, 1]

    x_progress = torch.clamp(box_x - prev_x, min=-0.05, max=0.05)
    y_center = -torch.abs(box_y - 1.2)
    yaw_reward = -yaw_err
    rotate_bonus = (yaw_err < 0.35).to(torch.float32)
    box_target_bonus = ((box_x > -1.4) & (box_x < 0.7) & (yaw_err < 0.45)).to(torch.float32)
    robot_progress = torch.clamp(robot_pos[:, 0] + 3.0, min=0.0, max=6.0)
    action_penalty = torch.sum(actions * actions, dim=-1)

    task_reward = env_reward.reshape(-1).to(args_cli.device, dtype=torch.float32)
    return (
        0.02 * task_reward
        + 6.0 * x_progress
        + 0.25 * yaw_reward
        + 0.05 * y_center
        + 0.5 * rotate_bonus
        + 2.0 * box_target_bonus
        + 0.02 * robot_progress
        - 0.002 * action_penalty
    )


def compute_gae(rewards, dones, values, last_value):
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros((args_cli.num_envs,), device=args_cli.device)
    for t in reversed(range(args_cli.horizon)):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == args_cli.horizon - 1 else values[t + 1]
        delta = rewards[t] + args_cli.gamma * next_value * next_nonterminal - values[t]
        last_advantage = delta + args_cli.gamma * args_cli.gae_lambda * next_nonterminal * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values
    return advantages, returns


def export_policy(model: ActorCritic, obs_mean: torch.Tensor, obs_std: torch.Tensor, out_path: str):
    class ExportedPolicy(nn.Module):
        def __init__(self, actor_critic: ActorCritic, mean: torch.Tensor, std: torch.Tensor):
            super().__init__()
            self.actor = actor_critic.actor
            self.register_buffer("command_scale", actor_critic.command_scale.cpu())
            self.register_buffer("mean", mean.cpu())
            self.register_buffer("std", std.cpu())

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            obs = (obs - self.mean) / self.std
            return torch.tanh(self.actor(obs)) * self.command_scale

    export = ExportedPolicy(model.cpu().eval(), obs_mean, obs_std).eval()
    example = torch.zeros((1, obs_mean.numel()), dtype=torch.float32)
    scripted = torch.jit.trace(export, example)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out))


def main():
    torch.manual_seed(args_cli.seed)
    os.makedirs(args_cli.ckpt_dir, exist_ok=True)

    env = make_env()
    low_level = LowLevelLocomotion(args_cli.device)
    obs, _ = env.reset()

    first_policy_obs, _robot_pos, box_pos = get_scene_tensors(env, obs)
    obs_dim = int(first_policy_obs.shape[-1])
    model = ActorCritic(obs_dim).to(args_cli.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args_cli.lr)

    obs_mean = torch.zeros(obs_dim, device=args_cli.device)
    obs_var = torch.ones(obs_dim, device=args_cli.device)
    obs_count = torch.tensor(1e-4, device=args_cli.device)

    prev_box_pos = box_pos.detach().clone()

    for iteration in range(1, args_cli.iterations + 1):
        obs_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, obs_dim), device=args_cli.device)
        raw_action_buf = torch.zeros((args_cli.horizon, args_cli.num_envs, 3), device=args_cli.device)
        logprob_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=args_cli.device)
        reward_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=args_cli.device)
        done_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=args_cli.device)
        value_buf = torch.zeros((args_cli.horizon, args_cli.num_envs), device=args_cli.device)

        episode_reward = 0.0
        for t in range(args_cli.horizon):
            policy_obs, robot_pos, box_pos = get_scene_tensors(env, obs)

            batch_mean = policy_obs.mean(dim=0)
            batch_var = policy_obs.var(dim=0, unbiased=False)
            batch_count = torch.tensor(policy_obs.shape[0], device=args_cli.device, dtype=torch.float32)
            delta = batch_mean - obs_mean
            total_count = obs_count + batch_count
            obs_mean = obs_mean + delta * batch_count / total_count
            obs_var = (
                obs_var * obs_count
                + batch_var * batch_count
                + delta * delta * obs_count * batch_count / total_count
            ) / total_count
            obs_count = total_count
            obs_std = torch.sqrt(obs_var).clamp_min(1e-4)

            norm_obs = (policy_obs - obs_mean) / obs_std
            with torch.no_grad():
                raw_action, command, logprob, value = model.act(norm_obs)
                env_action = low_level(obs, command)

            next_obs, env_reward, terminated, truncated, _info = env.step(env_action)
            next_policy_obs, next_robot_pos, next_box_pos = get_scene_tensors(env, next_obs)
            box_yaw = next_policy_obs[:, 14]
            done = (terminated | truncated).to(args_cli.device, dtype=torch.float32).reshape(-1)
            reward = shaped_reward(
                env,
                env_reward,
                next_robot_pos,
                next_box_pos,
                prev_box_pos,
                box_yaw,
                command,
            )

            obs_buf[t] = norm_obs
            raw_action_buf[t] = raw_action
            logprob_buf[t] = logprob
            reward_buf[t] = reward
            done_buf[t] = done
            value_buf[t] = value

            episode_reward += float(reward.mean().item())
            obs = next_obs
            prev_box_pos = next_box_pos.detach().clone()

            if bool((terminated | truncated).any().item()):
                obs, _ = env.reset()
                _policy_obs, _robot_pos, prev_box_pos = get_scene_tensors(env, obs)
                prev_box_pos = prev_box_pos.detach().clone()

        with torch.no_grad():
            last_policy_obs, _robot_pos, _box_pos = get_scene_tensors(env, obs)
            obs_std = torch.sqrt(obs_var).clamp_min(1e-4)
            last_value = model.critic((last_policy_obs - obs_mean) / obs_std).squeeze(-1)

        advantages, returns = compute_gae(reward_buf, done_buf, value_buf, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std().clamp_min(1e-6))

        flat_obs = obs_buf.reshape(-1, obs_dim)
        flat_actions = raw_action_buf.reshape(-1, 3)
        flat_logprobs = logprob_buf.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = value_buf.reshape(-1)

        batch_size = flat_obs.shape[0]
        minibatch_size = batch_size // args_cli.minibatches
        indices = torch.arange(batch_size, device=args_cli.device)

        policy_loss_value = 0.0
        value_loss_value = 0.0
        entropy_value = 0.0
        for _epoch in range(args_cli.epochs):
            perm = indices[torch.randperm(batch_size, device=args_cli.device)]
            for start in range(0, batch_size, minibatch_size):
                mb_idx = perm[start:start + minibatch_size]
                _command, new_logprob, entropy, new_value = model.evaluate(flat_obs[mb_idx], flat_actions[mb_idx])
                logratio = new_logprob - flat_logprobs[mb_idx]
                ratio = torch.exp(logratio)

                mb_adv = flat_advantages[mb_idx]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - args_cli.clip, 1.0 + args_cli.clip)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * (new_value - flat_returns[mb_idx]).pow(2).mean()
                entropy_loss = entropy.mean()
                loss = policy_loss + args_cli.value_coef * value_loss - args_cli.entropy_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args_cli.max_grad_norm)
                optimizer.step()

                policy_loss_value = float(policy_loss.item())
                value_loss_value = float(value_loss.item())
                entropy_value = float(entropy_loss.item())

        if iteration % args_cli.save_every == 0 or iteration == args_cli.iterations:
            ckpt_path = Path(args_cli.ckpt_dir) / f"iter_{iteration:05d}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_mean": obs_mean.detach().cpu(),
                    "obs_std": torch.sqrt(obs_var).clamp_min(1e-4).detach().cpu(),
                    "obs_dim": obs_dim,
                    "iteration": iteration,
                },
                ckpt_path,
            )
            export_policy(model, obs_mean.detach(), torch.sqrt(obs_var).clamp_min(1e-4).detach(), args_cli.out)

        print(
            f"[ppo] iter={iteration:04d} "
            f"reward={episode_reward / args_cli.horizon:.4f} "
            f"policy_loss={policy_loss_value:.4f} "
            f"value_loss={value_loss_value:.4f} "
            f"entropy={entropy_value:.4f}"
        )

    export_policy(model, obs_mean.detach(), torch.sqrt(obs_var).clamp_min(1e-4).detach(), args_cli.out)
    env.close()
    print(f"[ppo] exported {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()
