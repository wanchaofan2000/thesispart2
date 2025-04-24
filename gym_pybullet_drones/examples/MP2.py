"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
Swith two drones' postions

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

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.ACCcontrol import AccelerationControl
from gym_pybullet_drones.utils.Logger1 import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 6
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 25
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=6,
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
        [4, 4, 1], [-4, -4, 2],
        [5.2, 5.2, 1], [2.8, 2.8, 1],
        [-2.8, -2.8, 2], [-5.2, -5.2, 2]
    ])
    INIT_RPYS = np.array([[0, 0,  0] for i in range(num_drones)])
    FORMATION_OFFSETS = np.array([
        [0, 0, 0],
        [1.2, 1.2, 0],
        [-1.2, -1.2, 0]
    ])

    TARGET_POS_A = np.array([-4, -4, 1])
    TARGET_POS_B = np.array([4, 4, 1])
    TARGET_POS = np.vstack([
        TARGET_POS_A + FORMATION_OFFSETS[0], TARGET_POS_B + FORMATION_OFFSETS[0],
        TARGET_POS_A + FORMATION_OFFSETS[1], TARGET_POS_A + FORMATION_OFFSETS[2],
        TARGET_POS_B + FORMATION_OFFSETS[1], TARGET_POS_B + FORMATION_OFFSETS[2]
    ])


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
        ctrl = [AccelerationControl(drone_model=drone) for i in range(num_drones)] 

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    START = time.time()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)
        leader_num = 2

        leader_positions = np.array([obs[i][:3] for i in range(leader_num)])
        distances = pdist(leader_positions, metric='euclidean')
        min_distance = np.min(distances)


        # 26个方向向量
        possible_directions = np.array([
            [dx, dy, dz] for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1] if (dx, dy, dz) != (0, 0, 0)
        ])
        leader_indices = [0, 1]
        new_target_pos = np.zeros((num_drones, 3))

        leader_follower_map = {
            0: [2, 3],  # 0 号 leader 控制 2,3
            1: [4, 5]   # 1 号 leader 控制 4,5
        }

        new_target_acc = np.zeros((num_drones, 3))

        k1, k2, k3 = 20, -10, 0  # 权重系数

        for leader in leader_indices:
            current_pos = obs[leader][:3]  # 当前位置
            target_pos = TARGET_POS[leader]  # 目标点
            v_current = obs[leader][10:13]  # 当前速度

            max_min_distance = -np.inf  # 记录最佳方向的最小距离
            best_direction_score = -np.inf  # 记录最佳方向的分数
            step_time = 1 / env.CTRL_FREQ  # 时间步长
            acceleration = 5

            for direction in possible_directions:
                a_vec = acceleration * direction / np.linalg.norm(direction)  # 归一化加速度
                v_new = v_current + a_vec * step_time  # 计算新速度
                # 速度预测限幅检查，超过直接跳过该方向
                if np.linalg.norm(v_new) > 20.0:
                    continue  # 跳过这个方向
                predicted_position = current_pos + v_new * step_time  # 预测下一步位置

                # 计算新位置与其他无人机的最小距离
                temp_positions = leader_positions.copy()
                temp_positions[leader] = predicted_position  # 只更新当前无人机的位置
                new_distances = pdist(temp_positions, metric='euclidean')
                new_min_distance = np.min(new_distances)

                if new_min_distance <= 4.0:
                    collision_cost = math.exp(-2 * (new_min_distance - 4))  # exponential penalty
                else:
                    collision_cost = 0.0

                distance_to_target = np.linalg.norm(predicted_position - target_pos)

                # 计算到终点的距离
                distance_to_target = np.linalg.norm(predicted_position - target_pos)

                velocity_alignment_cost = max(0, np.dot(a_vec, v_current))  # 速度对齐成本

                # 总成本
                direction_score = k1 * collision_cost + k2 * distance_to_target + k3 * velocity_alignment_cost

                if direction_score > best_direction_score:
                    best_direction_score = direction_score
                    best_acc = a_vec

                new_target_acc[leader] = best_acc
                for follower in leader_follower_map[leader]:
                    new_target_acc[follower] = best_acc




        # 计算控制
        for j in range(num_drones):
            cur_pos = obs[j][:3]
            cur_quat = obs[j][3:7]
            cur_vel = obs[j][10:13]
            cur_ang_vel = obs[j][13:16]

            action[j, :], _ = ctrl[j].computeControl(
                control_timestep=env.CTRL_TIMESTEP,
                cur_pos=cur_pos,
                cur_quat=cur_quat,
                cur_vel=cur_vel,
                cur_ang_vel=cur_ang_vel,
                target_acc=new_target_acc[j],
                target_rpy=INIT_RPYS[j, :]
            )

        # 记录日志
        logger.log(
            drone=j,
            timestamp=i/env.CTRL_FREQ,
            state=obs[j],
            control=np.hstack([new_target_pos[j], INIT_RPYS[j, :], np.zeros(6)]),
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
