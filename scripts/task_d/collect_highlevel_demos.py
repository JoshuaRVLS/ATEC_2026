from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Collect Task D high-level command demonstrations.")
parser.add_argument("--task", type=str, default="ATEC-TaskD-B2Piper")
parser.add_argument("--output", type=str, default="datasets/task_d_highlevel/demos.hdf5")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--debug", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402
from demo.solution import AlgSolution  # noqa: E402
from scripts.task_d.highlevel_utils import COMMAND_DIM, FEATURE_DIM, build_feature  # noqa: E402


def make_env():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    return env


def collect_episode(env, episode_idx: int):
    solution = AlgSolution()
    obs, _ = env.reset()
    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else 0.02

    features: list[np.ndarray] = []
    commands: list[np.ndarray] = []
    scores: list[float] = []
    phases: list[str] = []

    total_score = 0.0
    elapsed = 0.0
    for step in range(args_cli.max_steps):
        if not simulation_app.is_running():
            break

        resp = solution.predicts(obs, total_score)
        if resp.get("giveup", False):
            break

        feature = build_feature(obs, solution, total_score, dt=float(dt), update_pose=False)
        cmd = solution.fixed_velocity_commands.detach().cpu().view(-1).numpy().astype(np.float32)
        if cmd.shape[0] != COMMAND_DIM:
            raise ValueError(f"Expected command dim {COMMAND_DIM}, got {cmd.shape[0]}")

        actions = torch.tensor(resp["action"], dtype=torch.float32, device=args_cli.device).view(1, -1)
        obs, reward, terminated, truncated, info = env.step(actions)

        sim_dt = info["Step_dt"]
        if isinstance(reward, torch.Tensor):
            total_score += reward.mean().item() / sim_dt
        else:
            total_score += float(reward) / sim_dt
        elapsed = info.get("Elapsed_Time", elapsed)
        elapsed = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)

        features.append(feature)
        commands.append(cmd)
        scores.append(float(total_score))
        phases.append(str(solution.phase))

        if args_cli.debug and step % 100 == 0:
            print(
                f"[collect] ep={episode_idx} step={step} "
                f"score={total_score:.2f} phase={solution.phase} cmd={cmd.tolist()}"
            )

        if bool(terminated.item() or truncated.item()):
            break

        if args_cli.real_time and dt is not None:
            time.sleep(float(dt))

    return {
        "features": np.asarray(features, dtype=np.float32).reshape(-1, FEATURE_DIM),
        "commands": np.asarray(commands, dtype=np.float32).reshape(-1, COMMAND_DIM),
        "scores": np.asarray(scores, dtype=np.float32),
        "phases": np.asarray(phases, dtype=h5py.string_dtype(encoding="utf-8")),
        "final_score": float(total_score),
        "elapsed": float(elapsed),
    }


def main():
    output_path = Path(args_cli.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env()
    with h5py.File(output_path, "w") as h5:
        h5.attrs["task"] = args_cli.task
        h5.attrs["feature_dim"] = FEATURE_DIM
        h5.attrs["command_dim"] = COMMAND_DIM

        for episode_idx in range(args_cli.episodes):
            data = collect_episode(env, episode_idx)
            group = h5.create_group(f"episode_{episode_idx:04d}")
            group.create_dataset("features", data=data["features"], compression="gzip")
            group.create_dataset("commands", data=data["commands"], compression="gzip")
            group.create_dataset("scores", data=data["scores"], compression="gzip")
            group.create_dataset("phases", data=data["phases"])
            group.attrs["final_score"] = data["final_score"]
            group.attrs["elapsed"] = data["elapsed"]
            print(
                f"[collect] saved episode_{episode_idx:04d}: "
                f"steps={len(data['features'])}, final_score={data['final_score']:.2f}"
            )

    env.close()
    print(f"[collect] wrote {output_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
