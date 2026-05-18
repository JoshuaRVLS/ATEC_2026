import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="View ATEC Task E.")
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--save_cam",
    action="store_true",
    default=False,
    help="Save Task E camera views to disk while the viewer is running.",
)
parser.add_argument(
    "--cam_dir",
    type=str,
    default="debug_cam",
    help="Directory where camera frames are written.",
)
parser.add_argument(
    "--save_every",
    type=int,
    default=10,
    help="Write one frame every N simulation steps.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from atec_rl_lab.tasks.task_e.env_cfg import TaskEEnvPiperCfg


def _to_rgb_uint8(frame: torch.Tensor) -> np.ndarray:
    """Convert camera output to HWC uint8 RGB."""
    image = frame.detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] in (3, 4):
        image = image[:3].permute(1, 2, 0)
    elif image.shape[-1] in (3, 4):
        image = image[..., :3]
    else:
        raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}")

    if image.dtype != torch.uint8:
        image = (image.float().clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return image.numpy()


def _save_camera_frame(camera, output_path: Path) -> None:
    """Write a single RGB frame to disk."""
    from PIL import Image

    frame = _to_rgb_uint8(camera.data.output["rgb"][0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)


def main():
    env_cfg = TaskEEnvPiperCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(env_cfg)
    output_dir = Path(args_cli.cam_dir)

    for name, articulation in env.scene.articulations.items():
        print("-" * 100)
        print("Robot name:", name)
        print("Bodies:", articulation.num_bodies, "->", articulation.body_names)
        print("Joints:", articulation.num_joints, "->", articulation.joint_names)
        articulation.set_joint_position_target(articulation.data.default_joint_pos)

    action_space = env.action_space
    obs, info = env.reset()
    video_cam = env.scene["video_cam"]
    ee_cam = env.scene["ee_camera"]

    for i in itertools.count():
        if not simulation_app.is_running():
            break
        action = torch.zeros(action_space.shape, device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)

        if args_cli.save_cam and i % max(args_cli.save_every, 1) == 0:
            _save_camera_frame(video_cam, output_dir / "video_cam" / f"frame_{i:06d}.png")
            _save_camera_frame(ee_cam, output_dir / "ee_camera" / f"frame_{i:06d}.png")

        done = terminated | truncated

        if done.any():
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            env.reset(env_ids=env_ids)


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
