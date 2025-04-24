"""

目前版本已经加了收缩因子，但是加入后因为判断增加会明显降低仿真速度

"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import sys
new_path = 'C:\deeprealm\python_robotic\gym-pybullet-drones'
sys.path.append(new_path)
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation as R

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

    # 为每条路径扩展k维度（如 channel：0.9, 1.0, 1.1）
    k_values = [0.9, 1.0, 1.1]
    full_library = []
    for path in base_paths:
        for k in k_values:
            full_library.append({
                'path': path.copy(),
                'k': k
            })

    return full_library

path_library = generate_path_library()

def compute_aligned_endpoints(current_pos, current_vel, path):
    """
    输入：当前位置 current_pos、当前速度 current_vel、路径原语库 path_library
    输出：所有对齐并平移后的轨迹终点位置组成的 endpoints 数组
    """
    v_norm = np.linalg.norm(current_vel)
    v_dir = np.array([1.0, 0.0, 0.0]) if v_norm < 1e-6 else current_vel / v_norm

    x_axis = np.array([1.0, 0.0, 0.0])

    if np.allclose(v_dir, x_axis):
        rot_matrix = np.eye(3)
    else:
        axis = np.cross(x_axis, v_dir)
        angle = np.arccos(np.clip(np.dot(x_axis, v_dir), -1.0, 1.0))
        axis /= np.linalg.norm(axis)
        rot_matrix = R.from_rotvec(angle * axis).as_matrix()

    endpoint = np.array([
        (rot_matrix @ path.T).T[-1] + current_pos
    ])

    return endpoint

paths = generate_path_library()

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 2
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 10
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=2,
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
    #### Initialize the simulation #############################
    INIT_XYZS = np.array([
        [4, 4, 1], [-4, -4, 1.1]
    ])
    INIT_RPYS = np.array([[0, 0,  0] for i in range(num_drones)])

    TARGET_POS_A = np.array([-4, -4, 1])
    TARGET_POS_B = np.array([4, 4, 1])
    TARGET_POS = np.vstack([
        TARGET_POS_A , TARGET_POS_B])


    #### Debug trajectory ######################################
    #### Uncomment alt. target_pos in .computeControlFromState()
    # INIT_XYZS = np.array([[.3 * i, 0, .1] for i in range(num_drones)])
    # INIT_RPYS = np.array([[0, 0,  i * (np.pi/3)/num_drones] for i in range(num_drones)])
    # NUM_WP = control_freq_hz*15
    # TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     if i < NUM_WP/6:
    #         TARGET_POS[i, :] = (i*6)/NUM_WP, 0, 0.5*(i*6)/NUM_WP
    #     elif i < 2 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-NUM_WP/6)*6)/NUM_WP, 0, 0.5 - 0.5*((i-NUM_WP/6)*6)/NUM_WP
    #     elif i < 3 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, ((i-2*NUM_WP/6)*6)/NUM_WP, 0.5*((i-2*NUM_WP/6)*6)/NUM_WP
    #     elif i < 4 * NUM_WP/6:
    #         TARGET_POS[i, :] = 0, 1 - ((i-3*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-3*NUM_WP/6)*6)/NUM_WP
    #     elif i < 5 * NUM_WP/6:
    #         TARGET_POS[i, :] = ((i-4*NUM_WP/6)*6)/NUM_WP, ((i-4*NUM_WP/6)*6)/NUM_WP, 0.5*((i-4*NUM_WP/6)*6)/NUM_WP
    #     elif i < 6 * NUM_WP/6:
    #         TARGET_POS[i, :] = 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 1 - ((i-5*NUM_WP/6)*6)/NUM_WP, 0.5 - 0.5*((i-5*NUM_WP/6)*6)/NUM_WP
    # wp_counters = np.array([0 for i in range(num_drones)])

    #### Create the environment ################################
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
                        user_debug_gui=user_debug_gui
                        )

    #### Obtain the PyBullet Client ID from the environment ####
    time.sleep(1)
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)] 

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        positions = np.array([obs[i][:3] for i in range(num_drones)])
        distances = pdist(positions, metric='euclidean')
        min_distance = np.min(distances)
        best_goals = np.zeros((num_drones, 3))

        lambda_g = 1.0     # 靠近目标的代价系数
        lambda_c = 10.0    # 碰撞代价权重（应远大于 lambda_g）
        d_safe = 1.2      # 安全半径
        beta = -2        # 距离过近时惩罚指数的陡峭度

        for drone in range(num_drones):
            current_pos = obs[drone][:3]
            current_vel = obs[drone][10:13]
            target_pos = TARGET_POS[drone]
            direction_to_target = target_pos - current_pos
            distance_to_target = np.linalg.norm(direction_to_target)
            t_now = i * env.CTRL_TIMESTEP

            # 如果距离足够近，直接目标点控制
            if distance_to_target < 0.5:
                best_goals[drone] = target_pos
            else:
                # 判断当前时间是否处于初始化阶段（如0.4秒前）
                if t_now < 0.4:
                    align_dir = direction_to_target / np.linalg.norm(direction_to_target)
                else:
                    v_norm = np.linalg.norm(current_vel)

                    align_dir = current_vel / v_norm

                endpoints = []
                goal_dists = []
                min_dists = []
                shrink_penalties = []
                collision_penalties = []
                total_costs = []
                others = np.delete(positions, drone, axis=0)
                for p in path_library:
                    k = p['k']
                    path = p['path']
                    endpoint = compute_aligned_endpoints(current_pos, align_dir, path)
                    endpoints.append(endpoint)
                    goal_dists.append(np.linalg.norm(endpoint - target_pos))
                    # 最近他人距离
                    dist_to_others = np.min(np.linalg.norm(endpoint - others, axis=1))
                    min_dists.append(dist_to_others)

                    # 非线性避碰代价（带收缩因子k）
                    k_d_safe = k * d_safe
                    if dist_to_others < k_d_safe:
                        collision_penalty = np.exp(-beta * (dist_to_others - k_d_safe)**2)
                    else:
                        collision_penalty = 0.0
                    collision_penalties.append(collision_penalty)

                    # 收缩惩罚
                    gamma = 1
                    shrink_penalty = np.exp(gamma * (k - 1)**2)
                    shrink_penalties.append(shrink_penalty)

                    total_costs.append(lambda_g * goal_dists[-1] + lambda_c * collision_penalty + shrink_penalty)

                    best_idx = np.argmin(total_costs)
                best_goal = endpoints[best_idx]
                best_k = path_library[best_idx]['k']
                best_goals[drone] = best_goal


            # 计算控制
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=best_goals[j],  # 设定新的目标点
                target_rpy=INIT_RPYS[j, :]
            )

            # 记录日志
            logger.log(
                drone=j,
                timestamp=i/env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([best_goals[j], INIT_RPYS[j, :], np.zeros(6)]),
            )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()



    #### Plot the simulation results ###########################
    if plot:
        logger.plot()

if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Helix flight script using CtrlAviary and DSLPIDControl')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model (default: CF2X)', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,          type=int,           help='Number of drones (default: 3)', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,      type=Physics,       help='Physics updates (default: PYB)', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,       type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,      type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Whether to plot the simulation results (default: True)', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,      type=str2bool,      help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,       type=str2bool,      help='Whether to add obstacles to the environment (default: True)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ,        type=int,           help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,         type=int,           help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,         type=int,           help='Duration of the simulation in seconds (default: 5)', metavar='')
    parser.add_argument('--output_folder',     default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,           help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
