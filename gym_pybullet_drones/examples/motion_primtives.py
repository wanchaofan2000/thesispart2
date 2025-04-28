### motion_primtives.py (润色整理版)

"""
8无人机编队运动原语选择程序
- 2个领导者（0, 4）主动选取运动原语
- 领导者收缩因子k影响自身和从机的布局
- 从机围绕领导者保持等边三角形跟随
- 动态维护每个领导者自己的d_safe

更新记录：
- 加入收缩因子k（0.9, 1.0, 1.1）
- 每步根据best_k动态更新d_safe
- 仅在领导者间计算避障代价
- 统一结构，整理逻辑，标准化缩进
"""

import os
import sys
import time
import math
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from datetime import datetime
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R

new_path = 'C:\\deeprealm\\python_robotic\\gym-pybullet-drones'
sys.path.append(new_path)

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger1 import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


def generate_path_library(
    arc_lengths=[1],
    arc_radii=[np.inf, 78.0, 36.0, 20.0, 12.0, 8.0, 6.0, 4.0, 3.0],
    delta_angle_deg=30
):
    """生成基本运动原语库，每条路径附加k维度。"""
    base_paths = []
    for radius in arc_radii:
        for length in arc_lengths:
            if math.isinf(radius):
                x = np.linspace(0, length, 10)
                y = np.zeros_like(x)
                z = np.zeros_like(x)
                path = np.stack([x, y, z], axis=1)
                base_paths.append(path)
            else:
                theta_max = length / radius
                thetas = np.linspace(0, theta_max, 10)
                x = radius * np.sin(thetas)
                y = radius * (1 - np.cos(thetas))
                z = np.zeros_like(thetas)
                arc = np.stack([x, y, z], axis=1)
                for angle in range(0, 360, delta_angle_deg):
                    rot = R.from_euler('x', angle, degrees=True).as_matrix()
                    rotated = (rot @ arc.T).T
                    base_paths.append(rotated)

    # 扩展k维度：0.9、1.0、1.1
    k_values = [0.9, 1.0, 1.1]
    full_library = []
    for path in base_paths:
        for k in k_values:
            full_library.append({
                'path': path.copy(),
                'k': k
            })
    return full_library

# 生成原语库
path_library = generate_path_library()


### 默认参数定义
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 8
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 5
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


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
    """主仿真入口"""
    #### 1. 初始化无人机位置与目标点 #######################
    center_positions = np.array([
        [4, 4, 2.0],
        [-4, -4, 1.8]
    ])

    triangle_offset = 0.8
    sqrt3_over_2 = np.sqrt(3)/2

    relative_offsets = np.array([
        [0.0, 0.0, 0.0],
        [triangle_offset, 0.0, 0.0],
        [-triangle_offset/2, sqrt3_over_2*triangle_offset, 0.0],
        [-triangle_offset/2, -sqrt3_over_2*triangle_offset, 0.0]
    ])

    INIT_XYZS = np.vstack([
        center_positions[0, :] + relative_offsets,
        center_positions[1, :] + relative_offsets
    ])

    INIT_RPYS = np.zeros((num_drones, 3))

    TARGET_POS_CENTERS = np.array([
        [-4, -4, 1],
        [4, 4, 1]
    ])

    leader_indices = [0, 4]
    follower_indices = {0: [1, 2, 3], 4: [5, 6, 7]}
    leader_d_safes = np.ones(len(leader_indices)) * 1.5
    leader_d_safes_init = leader_d_safes

    base_offsets = np.array([
        [0, 0, 0],
        [triangle_offset, 0, 0],
        [-triangle_offset/2, sqrt3_over_2 * triangle_offset, 0],
        [-triangle_offset/2, -sqrt3_over_2 * triangle_offset, 0]
    ])

    #### 2. 创建环境、控制器、Logger #########################
    env = CtrlAviary(drone_model=drone,
                     num_drones=num_drones,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     pyb_freq=simulation_freq_hz,
                     ctrl_freq=control_freq_hz,
                     gui=gui,
                     record=record_video,
                     obstacles=obstacles,
                     user_debug_gui=user_debug_gui)

    time.sleep(1)
    PYB_CLIENT = env.getPyBulletClient()

    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab)

    ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    #### 3. 主仿真循环 ######################################
    action = np.zeros((num_drones, 4))
    START = time.time()

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)

        positions = np.array([obs[i][:3] for i in range(num_drones)])
        leader_positions = np.array([obs[i][:3] for i in leader_indices])

        best_goals = np.zeros((num_drones, 3))
        lambda_g, lambda_c, beta, gamma = 1.0, 1.0, -1.0, 0.0001

        for drone in leader_indices:
            idx = leader_indices.index(drone)
            current_pos = obs[drone][:3]
            current_vel = obs[drone][10:13]
            target_pos = TARGET_POS_CENTERS[drone // 4]
            direction_to_target = target_pos - current_pos
            distance_to_target = np.linalg.norm(direction_to_target)

            t_now = i * env.CTRL_TIMESTEP

            if distance_to_target < 0.5:
                best_goals[drone] = target_pos
            else:
                if t_now < 0.4:
                    align_dir = direction_to_target / np.linalg.norm(direction_to_target)
                else:
                    v_norm = np.linalg.norm(current_vel)
                    align_dir = current_vel / v_norm if v_norm > 1e-3 else direction_to_target / np.linalg.norm(direction_to_target)

                x_axis = np.array([1, 0, 0])
                if np.allclose(align_dir, x_axis):
                    rot_matrix = np.eye(3)
                else:
                    axis = np.cross(x_axis, align_dir)
                    axis /= np.linalg.norm(axis)
                    angle = np.arccos(np.clip(np.dot(x_axis, align_dir), -1.0, 1.0))
                    rot_matrix = R.from_rotvec(angle * axis).as_matrix()

                paths = np.array([p['path'] for p in path_library])
                ks = np.array([p['k'] for p in path_library])
                rotated_paths = paths @ rot_matrix.T
                endpoints = rotated_paths[:, -1, :] + current_pos

                goal_dists = np.linalg.norm(endpoints - target_pos, axis=1)

                other_leaders = [idx for idx in leader_indices if idx != drone]
                other_positions = np.array([obs[i][:3] for i in other_leaders])

                if len(other_positions) > 0:
                    dist_matrix = np.linalg.norm(endpoints[:, None, :] - other_positions[None, :, :], axis=2)
                    min_dists = np.min(dist_matrix, axis=1)
                else:
                    min_dists = np.ones((endpoints.shape[0],)) * 1e6

                k_d_safes = ks * leader_d_safes[idx]
                collision_penalties = np.where(min_dists < k_d_safes,
                                               np.exp(-beta * (min_dists - k_d_safes)**2),
                                               0.0)
                
                current_init_d_safe = leader_d_safes_init[idx]  # 对应领导者的初始安全半径
                current_d_safe = leader_d_safes[idx]             # 对应领导者当前安全半径

                shrink_ratios = current_d_safe * ks / current_init_d_safe  # (P,) 每条原语新d_safe / 初始d_safe

                # 收缩（shrink_ratios < 1）有惩罚，膨胀（shrink_ratios >= 1）无惩罚
                shrink_penalties = np.where(
                    shrink_ratios < 1,
                    np.exp(gamma * (1 - shrink_ratios) ** 2),
                    1.0
                )

                total_costs = lambda_g * goal_dists + lambda_c * collision_penalties + shrink_penalties

                best_idx = np.argmin(total_costs)
                best_goal = endpoints[best_idx]
                best_k = ks[best_idx]

                leader_d_safes[idx] *= best_k
                best_goals[drone] = best_goal

                for j, follower in enumerate(follower_indices[drone]):
                    offset = base_offsets[j+1] * best_k
                    best_goals[follower] = best_goal + offset

        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=best_goals[j],
                target_rpy=INIT_RPYS[j, :]
            )

            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([best_goals[j], INIT_RPYS[j, :], np.zeros(6)]))

        env.render()

        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### 4. 仿真结束 #######################################
    env.close()

    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi-drone formation with motion primitives')
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