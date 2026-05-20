from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


PHASES = (
    "BACK_UP",
    "MOVE_LEFT_TO_BOX_LANE",
    "CONTACT_BOX",
    "PUSH_BOX",
    "DETACH_FROM_BOX",
    "MOVE_LEFT_OF_BOX",
    "MOVE_FORWARD_BESIDE_BOX",
    "ROTATE_BOX_RIGHT",
    "PUSH_BOX_RIGHT",
    "CROSS",
)


FEATURE_DIM = 3 + 3 + 3 + 3 + 4 + len(PHASES)
COMMAND_DIM = 3


def phase_one_hot(phase: str) -> np.ndarray:
    out = np.zeros((len(PHASES),), dtype=np.float32)
    if phase in PHASES:
        out[PHASES.index(phase)] = 1.0
    return out


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def get_depth_image(obs: dict[str, Any]) -> np.ndarray | None:
    image_obs = obs.get("image", {})
    if not isinstance(image_obs, dict):
        return None

    depth = None
    for key in ("head_depth", "video_depth", "ee_depth"):
        if image_obs.get(key) is not None:
            depth = _to_numpy(image_obs[key])
            break
    if depth is None:
        return None

    if depth.ndim == 4:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        return None
    return depth.astype(np.float32, copy=False)


def depth_summary(obs: dict[str, Any]) -> np.ndarray:
    depth = get_depth_image(obs)
    if depth is None:
        return np.zeros((4,), dtype=np.float32)

    height, width = depth.shape
    row0, row1 = int(height * 0.30), int(height * 0.88)
    col0, col1 = int(width * 0.08), int(width * 0.92)
    roi = depth[row0:row1, col0:col1]

    valid = np.isfinite(roi) & (roi > 0.15) & (roi < 5.0)
    valid_depth = roi[valid]
    if valid_depth.size < 200:
        return np.zeros((4,), dtype=np.float32)

    near_depth = np.quantile(valid_depth, 0.18) + 0.25
    near_mask = valid & (roi <= near_depth)
    if int(near_mask.sum()) < 120:
        return np.zeros((4,), dtype=np.float32)

    ys, xs = np.where(near_mask)
    xs = xs.astype(np.float32) + float(col0)
    ys = ys.astype(np.float32) + float(row0)

    center_x = xs.mean()
    center_y = ys.mean()
    image_center_x = float(width - 1) * 0.5
    image_center_y = float(height - 1) * 0.5
    x_error = np.clip((center_x - image_center_x) / max(image_center_x, 1.0), -1.0, 1.0)
    y_error = np.clip((center_y - image_center_y) / max(image_center_y, 1.0), -1.0, 1.0)
    distance = float(np.median(roi[near_mask]))
    coverage = float(near_mask.sum()) / float(max(roi.size, 1))

    return np.array([x_error, y_error, distance, coverage], dtype=np.float32)


def update_pose_estimate(solution: Any, obs: dict[str, Any], dt: float = 0.02) -> np.ndarray:
    proprio = _to_numpy(obs["proprio"])
    if proprio is None:
        return np.array([-3.0, 0.0, 0.0], dtype=np.float32)
    proprio = proprio.reshape(proprio.shape[0], -1)

    if not hasattr(solution, "est_x"):
        solution.est_x = -3.0
    if not hasattr(solution, "est_y"):
        solution.est_y = 0.0
    if not hasattr(solution, "est_yaw"):
        solution.est_yaw = 0.0

    vx_body = float(proprio[0, 0])
    vy_body = float(proprio[0, 1])
    yaw_rate = float(proprio[0, 5])

    solution.est_yaw += yaw_rate * dt
    cos_yaw = math.cos(solution.est_yaw)
    sin_yaw = math.sin(solution.est_yaw)
    vx_world = cos_yaw * vx_body - sin_yaw * vy_body
    vy_world = sin_yaw * vx_body + cos_yaw * vy_body
    solution.est_x += vx_world * dt
    solution.est_y += vy_world * dt

    return np.array([solution.est_x, solution.est_y, solution.est_yaw], dtype=np.float32)


def build_feature(
    obs: dict[str, Any],
    solution: Any,
    current_score: float,
    dt: float = 0.02,
    update_pose: bool = True,
) -> np.ndarray:
    proprio = _to_numpy(obs["proprio"])
    if proprio is None:
        raise ValueError("obs must contain proprio")
    proprio = proprio.reshape(proprio.shape[0], -1).astype(np.float32, copy=False)

    base_lin_vel = proprio[0, 0:3]
    base_ang_vel = proprio[0, 3:6]
    projected_gravity = proprio[0, 9:12]
    if update_pose:
        pose = update_pose_estimate(solution, obs, dt=dt)
    else:
        pose = np.array(
            [
                float(getattr(solution, "est_x", -3.0)),
                float(getattr(solution, "est_y", 0.0)),
                float(getattr(solution, "est_yaw", 0.0)),
            ],
            dtype=np.float32,
        )
    depth = depth_summary(obs)
    phase = phase_one_hot(getattr(solution, "phase", ""))

    return np.concatenate(
        [
            base_lin_vel,
            base_ang_vel,
            projected_gravity,
            np.array([float(current_score)], dtype=np.float32),
            pose,
            depth,
            phase,
        ],
        axis=0,
    ).astype(np.float32, copy=False)
