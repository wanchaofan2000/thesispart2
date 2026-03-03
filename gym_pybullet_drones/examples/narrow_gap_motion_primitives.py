### narrow_gap_motion_primitives.py
"""
四机编队运动原语示例：带收缩因子的窄缝通行
- 1 个领导者（0号机），3 个从机
- 初始编队为等边三角形布局（含领导者在顶点）
- 编队需穿过位于原点附近的窄缝，两侧为大长方体障碍
- 根据候选运动原语选择最优方向，并动态调整收缩因子 k
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


def generate_path_library(
    arc_lengths=None,
    arc_radii=None,
    delta_angle_deg=30
):
    """生成运动原语库（包含收缩因子选项）。"""
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
                path = np.stack([x, y, z], axis=1)
                base_paths.append(path)
            else:
                theta_max = length / radius
                thetas = np.linspace(0, theta_max, 12)
                x = radius * np.sin(thetas)
                y = radius * (1 - np.cos(thetas))
                z = np.zeros_like(thetas)
                arc = np.stack([x, y, z], axis=1)
                for angle in range(0, 360, delta_angle_deg):
                    rot = R.from_euler('x', angle, degrees=True).as_matrix()
                    rotated = (rot @ arc.T).T
                    base_paths.append(rotated)

    k_values = [0.98, 1.0, 1.02]
    full_library = []
    for path in base_paths:
        for k in k_values:
            full_library.append({
                'path': path.copy(),
                'k': k
            })
    return full_library


path_library = generate_path_library()


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
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

OBSTACLE_POINTS = None
OBSTACLE_KD_TREE = None

def _compute_obstacle_penalties(rotated_paths,
                                current_pos,
                                candidate_d_safes,
                                kd_tree,
                                min_safe_radius):
    """基于领导者路径与障碍的最近距离惩罚。"""
    obstacle_buffer = 0.12
    lambda_obs = 6.0

    if kd_tree is None:
        return np.zeros(rotated_paths.shape[0])

    penalties = np.zeros(rotated_paths.shape[0])
    for idx in range(rotated_paths.shape[0]):
        path_world = rotated_paths[idx] + current_pos
        dists, _ = kd_tree.query(path_world, k=1)
        min_dist = float(np.min(dists))

        effective_radius = max(min_safe_radius, candidate_d_safes[idx])
        clearance = min_dist - effective_radius

        # Keep a small positive buffer due to surface point-cloud discretization.
        if clearance < obstacle_buffer:
            penalties[idx] = lambda_obs * (obstacle_buffer - clearance)
        else:
            penalties[idx] = 0.0

    return penalties


def _compute_alignment_direction(direction_to_target, current_vel, t_now):
    """根据时间与速度计算路径对齐方向。"""
    if t_now < 0.4:
        return direction_to_target / np.linalg.norm(direction_to_target)

    v_norm = np.linalg.norm(current_vel)
    if v_norm > 1e-3:
        return current_vel / v_norm
    return direction_to_target / np.linalg.norm(direction_to_target)


def _rotation_from_x_axis(target_direction):
    """计算从 x 轴对齐到目标方向的旋转矩阵。"""
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
                        ks,
                        leader_d_safe,
                        leader_d_safe_init,
                        min_d_safe,
                        max_d_safe):
    """从候选运动原语中选取下一目标点与收缩因子。"""
    success_threshold = 0.4
    shrink_penalty_weight = 0.4

    direction_to_target = center_goal - current_pos
    distance_to_target = np.linalg.norm(direction_to_target)

    if distance_to_target < success_threshold:
        return center_goal, 1.0

    t_now = step * env.CTRL_TIMESTEP
    align_dir = _compute_alignment_direction(direction_to_target, current_vel, t_now)
    rot_matrix = _rotation_from_x_axis(align_dir)

    rotated_paths = paths @ rot_matrix.T
    endpoints = rotated_paths[:, -1, :] + current_pos
    goal_dists = np.linalg.norm(endpoints - center_goal, axis=1)

    candidate_d_safes = np.clip(leader_d_safe * ks, min_d_safe, max_d_safe)
    shrink_ratios = np.clip(candidate_d_safes / leader_d_safe_init, 0.0, 1.0)
    shrink_penalties = shrink_penalty_weight * np.exp(-np.square(shrink_ratios))
    obstacle_penalties = _compute_obstacle_penalties(rotated_paths,
                                                     current_pos,
                                                     candidate_d_safes,
                                                     OBSTACLE_KD_TREE,
                                                     min_d_safe)
    total_costs = goal_dists + shrink_penalties + obstacle_penalties
    best_idx = np.argmin(total_costs)
    return endpoints[best_idx], ks[best_idx]


def _plot_shrink_histories(shrink_histories, leader_indices):
    """绘制领导者收缩比历史。"""
    if not any(shrink_histories[leader] for leader in leader_indices):
        return

    plt.figure()
    for leader in leader_indices:
        history = shrink_histories[leader]
        if history:
            times, ratios = zip(*history)
            plt.plot(times, ratios, label=f"Leader {leader}")
    plt.xlabel('Time [s]')
    plt.ylabel('Shrink ratio (d_safe / d_safe_init)')
    plt.title('Leader shrink ratios over time')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)


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
    colab=DEFAULT_COLAB
):
    """主仿真入口：四机编队穿缝。"""
    global OBSTACLE_POINTS, OBSTACLE_KD_TREE

    center_start = np.array([-6.0, 0.0, 1.5])
    center_goal = np.array([6.0, 0.0, 1.5])

    triangle_offset = 0.8
    sqrt3_over_2 = np.sqrt(3) / 2
    base_offsets = np.array([
        [0.0, 0.0, 0.0],
        [triangle_offset, 0.0, 0.0],
        [-triangle_offset / 2, sqrt3_over_2 * triangle_offset, 0.0],
        [-triangle_offset / 2, -sqrt3_over_2 * triangle_offset, 0.0]
    ])

    INIT_XYZS = center_start + base_offsets
    INIT_RPYS = np.zeros((num_drones, 3))

    leader_indices = [0]
    follower_indices = {0: [1, 2, 3]}
    leader_d_safes = np.ones(len(leader_indices)) * 1.3
    leader_d_safes_init = leader_d_safes.copy()
    min_d_safe = 0.45
    max_d_safe = 1.7
    shrink_histories = {leader: [] for leader in leader_indices}

    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=6,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui)

    time.sleep(1)
    PYB_CLIENT = env.getPyBulletClient()

    OBSTACLE_POINTS, OBSTACLE_KD_TREE = build_narrow_gap_kdtree(PYB_CLIENT)

    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=DEFAULT_CAMERA_DISTANCE,
            cameraYaw=0.0,
            cameraPitch=-89.0,
            cameraTargetPosition=DEFAULT_GAP_CENTER.tolist(),
            physicsClientId=PYB_CLIENT
        )

    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab)

    controllers = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    action = np.zeros((num_drones, 4))
    START = time.time()

    paths = np.array([p['path'] for p in path_library])
    ks = np.array([p['k'] for p in path_library])

    for step in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)
        best_goals = np.array([obs[i][:3] for i in range(num_drones)])

        for idx, drone_id in enumerate(leader_indices):
            current_pos = obs[drone_id][:3]
            current_vel = obs[drone_id][10:13]
            best_goal, best_k = _select_leader_goal(current_pos=current_pos,
                                                    current_vel=current_vel,
                                                    center_goal=center_goal,
                                                    step=step,
                                                    env=env,
                                                    paths=paths,
                                                    ks=ks,
                                                    leader_d_safe=leader_d_safes[idx],
                                                    leader_d_safe_init=leader_d_safes_init[idx],
                                                    min_d_safe=min_d_safe,
                                                    max_d_safe=max_d_safe)

            leader_d_safes[idx] = np.clip(leader_d_safes[idx] * best_k, min_d_safe, max_d_safe)
            ratio = leader_d_safes[idx] / leader_d_safes_init[idx]
            shrink_histories[drone_id].append((step / env.CTRL_FREQ, ratio))
            best_goals[drone_id] = best_goal

            for j, follower in enumerate(follower_indices[drone_id]):
                offset = base_offsets[j + 1] * ratio
                best_goals[follower] = best_goal + offset

        for j in range(num_drones):
            action[j, :], _, _ = controllers[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=best_goals[j],
                target_rpy=INIT_RPYS[j, :]
            )

            logger.log(drone=j,
                       timestamp=step / env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([best_goals[j], INIT_RPYS[j, :], np.zeros(6)]))

        env.render()
        if gui:
            sync(step, START, env.CTRL_TIMESTEP)

    env.close()
    if plot:
        _plot_shrink_histories(shrink_histories, leader_indices)
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Four-drone formation through narrow gap with motion primitives')
    parser.add_argument('--drone', default=DEFAULT_DRONES, type=DroneModel)
    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int)
    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics)
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool)
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool)
    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool)
    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool)
    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool)
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int)
    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int)
    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int)
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str)
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool)
    ARGS = parser.parse_args()

    run(**vars(ARGS))
