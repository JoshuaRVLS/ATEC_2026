# Created by skywoodsz on 2026/02/07.

import argparse
import os
import sys
import time
import json
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.solution import AlgSolution
solution = AlgSolution()

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play Atec Tasks (ENV only, no RL).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="Enable debug prints for per-step reward/time metrics.",
)
parser.add_argument(
    "--teleop-record",
    type=str,
    default=None,
    help="Drive with keyboard and save velocity commands to this JSON file.",
)
parser.add_argument(
    "--teleop-replay",
    type=str,
    default=None,
    help="Replay a JSON velocity command recording.",
)

# Isaac Sim / Kit args
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# If recording video, need cameras enabled in IsaacLab/Kit
if args_cli.video:
    args_cli.enable_cameras = True

# -----------------------------------------------------------------------------
# Launch Isaac Sim / Kit
# -----------------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports AFTER simulation_app is created (IsaacLab pattern)
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import atec_rl_lab.tasks  # noqa: F401, E402 (register your tasks)
from isaaclab_tasks.utils import parse_env_cfg
from rl_utils import camera_follow


class KeyboardVelocityTeleop:
    """Small WASD/QE velocity teleop using Isaac Sim keyboard events."""

    def __init__(self, vx_step=0.12, vy_step=0.12, wz_step=0.18, max_v=0.9, max_w=1.2):
        self.vx_step = vx_step
        self.vy_step = vy_step
        self.wz_step = wz_step
        self.max_v = max_v
        self.max_w = max_w
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.world_frame = False
        self._pressed = set()
        self._sub = None

        import carb.input
        import omni.appwindow

        self._carb_input = carb.input
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = carb.input.acquire_input_interface().subscribe_to_keyboard_events(
            keyboard, self._on_keyboard_event
        )
        print("[TELEOP] W/S=vx, A/D=vy, Q/E=yaw, Space=stop, R=world/body toggle")

    def _on_keyboard_event(self, event):
        key = getattr(event.input, "name", str(event.input)).split(".")[-1].upper()
        event_type = event.type
        if event_type == self._carb_input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(key)
            if key == "SPACE":
                self.vx = self.vy = self.wz = 0.0
            elif key == "R":
                self.world_frame = not self.world_frame
                mode = "world" if self.world_frame else "body"
                print(f"[TELEOP] frame={mode}")
        elif event_type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(key)
        return True

    @staticmethod
    def _clamp(value, lo, hi):
        return max(lo, min(hi, value))

    def update(self):
        if "W" in self._pressed:
            self.vx += self.vx_step
        if "S" in self._pressed:
            self.vx -= self.vx_step
        if "A" in self._pressed:
            self.vy += self.vy_step
        if "D" in self._pressed:
            self.vy -= self.vy_step
        if "Q" in self._pressed:
            self.wz += self.wz_step
        if "E" in self._pressed:
            self.wz -= self.wz_step

        if not ({"W", "S"} & self._pressed):
            self.vx *= 0.82
        if not ({"A", "D"} & self._pressed):
            self.vy *= 0.82
        if not ({"Q", "E"} & self._pressed):
            self.wz *= 0.75

        if abs(self.vx) < 0.02:
            self.vx = 0.0
        if abs(self.vy) < 0.02:
            self.vy = 0.0
        if abs(self.wz) < 0.02:
            self.wz = 0.0

        self.vx = self._clamp(self.vx, -self.max_v, self.max_v)
        self.vy = self._clamp(self.vy, -self.max_v, self.max_v)
        self.wz = self._clamp(self.wz, -self.max_w, self.max_w)
        return self.vx, self.vy, self.wz, self.world_frame


def play() -> tuple[float, float]:
    if args_cli.task is None:
        raise ValueError("Please provide --task, e.g. --task ATEC-TaskA-G1")
    if args_cli.teleop_record and args_cli.teleop_replay:
        raise ValueError("Use either --teleop-record or --teleop-replay, not both.")

    is_task_e = isinstance(args_cli.task, str) and args_cli.task.startswith("ATEC-TaskE")
    # -------------------------------------------------------------------------
    # Create env (plain Gym env)
    # -------------------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Convert MARL -> single agent if needed (kept from your original script)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # -------------------------------------------------------------------------
    # Optional: video wrapper
    # -------------------------------------------------------------------------
    if args_cli.video:
        # Put videos in ./logs/videos/play by default (edit as you like)
        video_kwargs = {
            "video_folder": os.path.abspath(os.path.join("logs", "videos", args_cli.task, "play")),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------
    obs, _ = env.reset()

    dt = env.unwrapped.step_dt if hasattr(env.unwrapped, "step_dt") else None
    timestep = 0
    teleop = None
    recording = []
    replay = None
    if args_cli.teleop_record:
        teleop = KeyboardVelocityTeleop()
        solution.clear_manual_command()
    if args_cli.teleop_replay:
        with open(args_cli.teleop_replay, "r", encoding="utf-8") as f:
            payload = json.load(f)
        replay = payload["commands"] if isinstance(payload, dict) else payload
        print(f"[TELEOP] replay loaded {len(replay)} commands from {args_cli.teleop_replay}")

    # -------------------------------------------------------------------------
    # Play loop
    # -------------------------------------------------------------------------
    total_episode_reward = 0.0
    total_elapsed_time = 0.0
    while simulation_app.is_running():
        with torch.inference_mode():
            start_time = time.time()

            if teleop is not None:
                vx, vy, wz, world_frame = teleop.update()
                solution.set_manual_command(vx, vy, wz, world_frame=world_frame)
                recording.append({
                    "vx": vx,
                    "vy": vy,
                    "wz": wz,
                    "world_frame": world_frame,
                })
                if timestep % 25 == 0:
                    frame = "W" if world_frame else "B"
                    print(f"[TELEOP]{timestep:<4} frame={frame} cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f})")
            elif replay is not None:
                if timestep >= len(replay):
                    print("[TELEOP] replay finished")
                    break
                cmd = replay[timestep]
                solution.set_manual_command(
                    cmd.get("vx", 0.0),
                    cmd.get("vy", 0.0),
                    cmd.get("wz", 0.0),
                    world_frame=cmd.get("world_frame", False),
                )
            else:
                solution.clear_manual_command()

            # ===== Your controller goes here =====
            resp = solution.predicts(obs, total_episode_reward)
            giveup = resp["giveup"]
            if giveup:
                break
            actions = resp["action"]
            actions = torch.tensor(actions, dtype=torch.float32, device='cuda').view(1, -1)
            obs, reward, terminated, truncated, info = env.step(actions)
            if not is_task_e:
                camera_follow(env)

            sim_dt = info["Step_dt"]
            if isinstance(reward, torch.Tensor):
                total_episode_reward += reward.mean().item() / sim_dt
            else:
                total_episode_reward += float(reward) / sim_dt

            if isinstance(info, dict) and "Elapsed_Time" in info:
                elapsed = info["Elapsed_Time"]  # simulation time from env as primary source
                total_elapsed_time = elapsed.item() if hasattr(elapsed, "item") else float(elapsed)
            elif dt is not None:
                total_elapsed_time += dt  # wall clock time as fallback

            if args_cli.debug:
                print(f"total_episode_reward:{total_episode_reward: .2f}")
                print(f"total_elapsed_time:{total_elapsed_time: .2f}")

            done = (terminated.item() or truncated.item())
            if done:
                print(f"[TERM] terminated={terminated.item()} truncated={truncated.item()}")
                print(f"[TERM] step={timestep} elapsed={total_elapsed_time:.2f}s")
                if isinstance(info, dict):
                    print(f"[TERM] info keys={list(info.keys())}")
                    log = info.get("log")
                    if log is not None:
                        print(f"[TERM] log keys={list(log.keys())}")
                        for k, v in log.items():
                            val = v.item() if hasattr(v, 'item') else float(v)
                            print(f"  {k} = {val}")
                break

            timestep += 1
            # If recording one video, exit after video_length steps
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            if args_cli.real_time and dt is not None:
                sleep_time = dt - (time.time() - start_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    env.close()
    if args_cli.teleop_record:
        out_path = Path(args_cli.teleop_record)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task": args_cli.task,
            "dt": dt,
            "commands": recording,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[TELEOP] saved {len(recording)} commands to {out_path}")

    return total_episode_reward, total_elapsed_time


if __name__ == "__main__":
    score, elapsed_time = play()
    print(f"score: {score:.2f}, elapsed_time: {elapsed_time:.2f} seconds")

    # Finally, close the simulation app
    print("Closing simulation app...")
    simulation_app.close()
