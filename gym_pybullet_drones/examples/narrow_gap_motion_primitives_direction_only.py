### narrow_gap_motion_primitives_direction_only.py
"""
四机编队窄缝对比实验：仅方向性 motion primitives（无缩放因子）。

与 `narrow_gap_motion_primitives.py` 对比点：
- 保留方向性原语选择（直线/弧线方向搜索）
- 移除编队收缩因子 k，队形尺寸固定
- 同样要求向目标点飞行，但在窄缝场景下更容易绕行或在时限内失败
"""

import time
import math
import argparse
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.examples.obstacles import (
    DEFAULT_GAP_CENTER,
    DEFAULT_CAMERA_DISTANCE,
    build_narrow_gap_kdtree,
)


def generate_directional_path_library(
    arc_lengths=None,
    arc_radii=None,
    delta_angle_deg=30
):
    """生成仅含方向变化的运动原语库（无缩放因子）。"""
    if arc_lengths is None:
        arc_lengths = [0.6]
    if arc_radii is None:
        arc_radii = [np.inf, 60.0, 30.0, 18.0, 12.0, 8.0, 6.0, 4.0]

    base_paths = []
    for radius in arc_radii:
        for length in arc_lengths:
            if math.isinf(radius):
                x = np.linspace(0, length, 12)
                y = np.zeros_like(x)
                z = np.zeros_like(x)
                base_paths.append(np.stack([x, y, z], axis=1))
            else:
                theta_max = length / radius
                thetas = np.linspace(0, theta_max, 12)
                x = radius * np.sin(thetas)
                y = radius * (1 - np.cos(thetas))
                z = np.zeros_like(thetas)
                arc = np.stack([x, y, z], axis=1)
                for angle in range(0, 360, delta_angle_deg):
                    rot = R.from_euler('x', angle, degrees=True).as_matrix()
                    base_paths.append((rot @ arc.T).T)
    return np.array(base_paths)


path_library = generate_directional_path_library()


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 4
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_LEADER_SAFE_RADIUS = 1.3
DEFAULT_SUCCESS_THRESHOLD = 0.4
DEFAULT_DETOUR_RATIO = 1.15

OBSTACLE_POINTS = None
OBSTACLE_KD_TREE = None


def _compute_obstacle_penalties(rotated_paths,
                                current_pos,
                                kd_tree,
                                leader_safe_radius,
                                lambda_obs=3.0):
    """基于固定安全半径计算路径障碍惩罚。"""
    if kd_tree is None:
        return np.zeros(rotated_paths.shape[0])

    penalties = np.zeros(rotated_paths.shape[0])
    for idx in range(rotated_paths.shape[0]):
        path_world = rotated_paths[idx] + current_pos
        dists, _ = kd_tree.query(path_world, k=1)
        min_dist = float(np.min(dists))
        clearance = min_dist - leader_safe_radius
        penalties[idx] = lambda_obs * (-clearance) if clearance < 0 else 0.0
    return penalties


def _compute_alignment_direction(direction_to_target, current_vel, t_now):
    if t_now < 0.4:
        return direction_to_target / np.linalg.norm(direction_to_target)

    v_norm = np.linalg.norm(current_vel)
    if v_norm > 1e-3:
        return current_vel / v_norm
    return direction_to_target / np.linalg.norm(direction_to_target)


def _rotation_from_x_axis(target_direction):
    x_axis = np.array([1.0, 0.0, 0.0])
    if np.allclose(target_direction, x_axis):
        return np.eye(3)

    axis = np.cross(x_axis, target_direction)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-6:
        return np.eye(3)

    axis /= axis_norm
    angle = np.arccos(np.clip(np.dot(x_axis, target_direction), -1.0, 1.0))
    return R.from_rotvec(angle * axis).as_matrix()


def _select_leader_goal(current_pos,
                        current_vel,
                        center_goal,
                        step,
                        env,
                        paths,
                        leader_safe_radius):
    """仅方向原语选择下一目标点（无编队缩放）。"""
    direction_to_target = center_goal - current_pos
    distance_to_target = np.linalg.norm(direction_to_target)
    if distance_to_target < DEFAULT_SUCCESS_THRESHOLD:
        return center_goal

    t_now = step * env.CTRL_TIMESTEP
    align_dir = _compute_alignment_direction(direction_to_target, current_vel, t_now)
    rot_matrix = _rotation_from_x_axis(align_dir)

    rotated_paths = paths @ rot_matrix.T
    endpoints = rotated_paths[:, -1, :] + current_pos
    goal_dists = np.linalg.norm(endpoints - center_goal, axis=1)
    obstacle_penalties = _compute_obstacle_penalties(rotated_paths,
                                                     current_pos,
                                                     OBSTACLE_KD_TREE,
                                                     leader_safe_radius)
    total_costs = goal_dists + obstacle_penalties
    best_idx = np.argmin(total_costs)
    return endpoints[best_idx]


def _path_length(path_points):
    if len(path_points) < 2:
        return 0.0
    diffs = np.diff(np.array(path_points), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _plot_goal_distance(goal_distance_history):
    if not goal_distance_history:
        return
    ts, ds = zip(*goal_distance_history)
    plt.figure()
    plt.plot(ts, ds, label="Leader distance to goal")
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.title("Direction-only experiment: distance to goal")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()


def run(
    drone=DEFAULT_DRONES,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VISION,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    obstacles=DEFAULT_OBSTACLES,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB,
):
    """主仿真入口：仅方向原语的窄缝对比实验。"""
    global OBSTACLE_POINTS, OBSTACLE_KD_TREE

    center_start = np.array([-6.0, 0.0, 1.5])
    center_goal = np.array([6.0, 0.0, 1.5])

    triangle_offset = 0.8
    sqrt3_over_2 = np.sqrt(3) / 2
    base_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [triangle_offset, 0.0, 0.0],
            [-triangle_offset / 2, sqrt3_over_2 * triangle_offset, 0.0],
            [-triangle_offset / 2, -sqrt3_over_2 * triangle_offset, 0.0],
        ]
    )

    init_xyzs = center_start + base_offsets
    init_rpys = np.zeros((num_drones, 3))

    leader_indices = [0]
    follower_indices = {0: [1, 2, 3]}

    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=physics,
        neighbourhood_radius=6,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui,
    )

    time.sleep(1)
    pyb_client = env.getPyBulletClient()
    OBSTACLE_POINTS, OBSTACLE_KD_TREE = build_narrow_gap_kdtree(pyb_client)

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=DEFAULT_CAMERA_DISTANCE,
            cameraYaw=0.0,
            cameraPitch=-89.0,
            cameraTargetPosition=DEFAULT_GAP_CENTER.tolist(),
            physicsClientId=pyb_client,
        )

    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )
    controllers = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    action = np.zeros((num_drones, 4))
    start_wall_time = time.time()

    leader_positions = []
    goal_distance_history = []

    last_obs = None
    for step in range(int(duration_sec * env.CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)
        last_obs = obs
        best_goals = np.array([obs[i][:3] for i in range(num_drones)])

        leader_pos = obs[0][:3]
        leader_positions.append(leader_pos.copy())
        goal_distance_history.append((step / env.CTRL_FREQ, float(np.linalg.norm(center_goal - leader_pos))))

        for leader in leader_indices:
            current_pos = obs[leader][:3]
            current_vel = obs[leader][10:13]
            best_goal = _select_leader_goal(current_pos=current_pos,
                                            current_vel=current_vel,
                                            center_goal=center_goal,
                                            step=step,
                                            env=env,
                                            paths=path_library,
                                            leader_safe_radius=DEFAULT_LEADER_SAFE_RADIUS)
            best_goals[leader] = best_goal

            for j, follower in enumerate(follower_indices[leader]):
                best_goals[follower] = best_goal + base_offsets[j + 1]

        for j in range(num_drones):
            action[j, :], _, _ = controllers[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=best_goals[j],
                target_rpy=init_rpys[j, :],
            )
            logger.log(
                drone=j,
                timestamp=step / env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([best_goals[j], init_rpys[j, :], np.zeros(6)]),
            )

        env.render()
        if gui:
            sync(step, start_wall_time, env.CTRL_TIMESTEP)

    if last_obs is None:
        final_leader_pos = center_start.copy()
    else:
        final_leader_pos = np.array(last_obs[0][:3])
    final_dist = float(np.linalg.norm(center_goal - final_leader_pos))
    reached = final_dist < DEFAULT_SUCCESS_THRESHOLD
    traveled = _path_length(leader_positions)
    straight = float(np.linalg.norm(center_goal - center_start))
    detour = traveled > DEFAULT_DETOUR_RATIO * straight

    env.close()

    if plot:
        _plot_goal_distance(goal_distance_history)
        logger.plot()

    print("\n=== Direction-only motion primitives summary ===")
    print(f"Reached goal (<{DEFAULT_SUCCESS_THRESHOLD:.2f} m): {reached}")
    print(f"Final leader distance to goal: {final_dist:.3f} m")
    print(f"Leader traveled length: {traveled:.3f} m (straight line: {straight:.3f} m)")
    print(f"Detour detected (>{DEFAULT_DETOUR_RATIO:.2f}x straight): {detour}")
    if not reached:
        print("Result: failure within time limit is expected in narrow-gap case without shrink factor.")
    elif detour:
        print("Result: reached goal but with clear detour due to fixed formation size.")
    else:
        print("Result: reached goal with limited detour in this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Four-drone narrow-gap comparison: direction-only motion primitives (no scaling factor)"
    )
    parser.add_argument("--drone", default=DEFAULT_DRONES, type=DroneModel)
    parser.add_argument("--num_drones", default=DEFAULT_NUM_DRONES, type=int)
    parser.add_argument("--physics", default=DEFAULT_PHYSICS, type=Physics)
    parser.add_argument("--gui", default=DEFAULT_GUI, type=str2bool)
    parser.add_argument("--record_video", default=DEFAULT_RECORD_VISION, type=str2bool)
    parser.add_argument("--plot", default=DEFAULT_PLOT, type=str2bool)
    parser.add_argument("--user_debug_gui", default=DEFAULT_USER_DEBUG_GUI, type=str2bool)
    parser.add_argument("--obstacles", default=DEFAULT_OBSTACLES, type=str2bool)
    parser.add_argument("--simulation_freq_hz", default=DEFAULT_SIMULATION_FREQ_HZ, type=int)
    parser.add_argument("--control_freq_hz", default=DEFAULT_CONTROL_FREQ_HZ, type=int)
    parser.add_argument("--duration_sec", default=DEFAULT_DURATION_SEC, type=int)
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument("--colab", default=DEFAULT_COLAB, type=bool)
    args = parser.parse_args()

    run(**vars(args))
