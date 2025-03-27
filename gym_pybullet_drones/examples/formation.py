"""Script demonstrating the joint use of simulation and control.

The simulation is run by a `CtrlAviary` environment.
The control is given by the PID implementation in `DSLPIDControl`.

Example
-------
In a terminal, run as:

    $ python pid.py

Notes
-----
The drones move, at different altitudes, along cicular trajectories 
in the X-Y plane, around point (0, -.3).

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
from network import Actor_GNN
import torch
from param import *

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.ACCcontrol import AccelerationControl
from gym_pybullet_drones.utils.Logger1 import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 3
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor = Actor_GNN(state_dim, 16 ,action_dim).to(device)
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "actor_gnn_32drones_1.pkl")
actor.load_network(model_path)

def get_gnn_state(obs_all, i, leader_id=0, Rc=5, init_obs_all=None):
    pos_i, vel_i = obs_all[i][0:2], obs_all[i][10:12]
    pos_0, vel_0 = obs_all[leader_id][0:2], obs_all[leader_id][10:12]

    # 当前无人机相对 leader 当前状态
    rel_state = np.concatenate([pos_i - pos_0, vel_i - vel_0])

    # 初始状态差
    init_pos_i, init_vel_i = init_obs_all[i][0:2], init_obs_all[i][10:12]
    init_pos_0, init_vel_0 = init_obs_all[leader_id][0:2], init_obs_all[leader_id][10:12]
    init_state = np.concatenate([init_pos_i - init_pos_0, init_vel_i - init_vel_0])

    state_list = [rel_state, init_state]

    # 邻居状态
    for j in range(obs_all.shape[0]):
        if j != i and j != leader_id:
            dist = np.linalg.norm(obs_all[j][0:2] - pos_i)
            if dist < Rc:
                rel_pos = obs_all[j][0:2] - pos_i
                rel_vel = obs_all[j][10:12] - vel_i
                state_list.append(np.concatenate([rel_pos, rel_vel]))

    return np.array(state_list)


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
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .6
    INIT_XYZS = np.array([[R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(num_drones)])
    INIT_RPYS = np.array([[0, 0,  i * (np.pi/2)/num_drones] for i in range(num_drones)])

    #### Initialize trajectory using way points ######################
    PERIOD = 10
    NUM_WP = control_freq_hz*PERIOD
    TARGET_POS = np.zeros((NUM_WP,3))
    # for i in range(NUM_WP):
    #     TARGET_POS[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0, 1], 0
    # wp_counters = np.array([int((i*NUM_WP/6)%NUM_WP) for i in range(num_drones)])

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
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(logging_freq_hz=control_freq_hz,
                    num_drones=num_drones,
                    output_folder=output_folder,
                    colab=colab
                    )

    #### Initialize the controllers ############################
    leaderctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    followerctrl = [AccelerationControl(drone_model=DroneModel.CF2X) for i in range(1,num_drones)]

    #### Run the simulation ####################################
    action = np.zeros((num_drones,4))
    omega = math.pi / 4
    START = time.time()
    z_integrals = np.zeros(num_drones)
    P_Z, I_Z, D_Z = 1.25, 0.05, 0.5 
    init_obs_all = env._computeObs()
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):

        current_time = time.time() - START
        obs, reward, terminated, truncated, info = env.step(action)
        leadertarget = np.zeros(3)
        leadertarget = R*np.sin(omega * current_time),  2*R*np.cos(omega * current_time) - 2*R, current_time/8 + 0.4 #现在这种轨迹模式用绝对时间，训练时候不能这样
        for j in range(num_drones):
            if j ==0:
                action[0, :], _, _ = leaderctrl.computeControlFromState(control_timestep=env.CTRL_TIMESTEP,
                                                             state = obs[0],
                                                             target_pos=leadertarget,
                                                             target_rpy=INIT_RPYS[0])
            else:
                state_i = get_gnn_state(obs, j, init_obs_all=init_obs_all)
                state_tensor = torch.tensor(state_i, dtype=torch.float32, device=device)
                with torch.no_grad():
                    acc = actor(state_tensor).cpu().numpy() 
                target_acc_xy = np.array([acc[0], acc[1]])

                # z轴 PID 控制项
                cur_z = obs[j][2]
                cur_z_dot = obs[j][11]
                target_z =  obs[0][2]

                # PID 控制项
                z_err = target_z - cur_z
                z_dot_err = -cur_z_dot
                z_integrals[j] += z_err * env.CTRL_TIMESTEP
                z_integrals[j] = np.clip(z_integrals[j], -1.0, 1.0)  # 限制积分发散

                acc_z = P_Z * z_err + D_Z * z_dot_err + I_Z * z_integrals[j]

                # 合成 target_acc
                target_acc = np.array([acc[0], acc[1], acc_z]) 
                action[j, :], _ = followerctrl[j-1].computeControl(
                    control_timestep=env.CTRL_TIMESTEP,
                    cur_pos=obs[j][0:3],
                    cur_quat=obs[j][3:7],
                    cur_vel=obs[j][10:13],
                    cur_ang_vel=obs[j][13:16],
                    target_acc=target_acc,
                    target_rpy=np.array([0, 0, 0])
                )



        #### Compute control for the current way point #############
        


        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(drone=j,
                       timestamp=i/env.CTRL_FREQ,
                       state=obs[j],
                       control=np.hstack([leadertarget, INIT_RPYS[j, :], np.zeros(6)])
                       # control=np.hstack([INIT_XYZS[j, :]+TARGET_POS[wp_counters[j], :], INIT_RPYS[j, :], np.zeros(6)])
                       )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("pid") # Optional CSV save

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
