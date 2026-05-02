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
from gym_pybullet_drones.examples.plot_record_utils import (
    DEFAULT_LOG_DIR,
    get_obstacle_box_specs,
    plot_3d_trajectories,
    plot_safety_histories,
    plot_top_view_trajectories,
    resolve_output_path,
    save_summary_text,
    save_thesis_figure,
)


def generate_directional_path_library(
    arc_lengths=None,
    arc_radii=None,
    delta_angle_deg=30,
    lateral_offsets=None,
):
    """生成仅含方向变化的运动原语库（无缩放因子）。"""
    if arc_lengths is None:
        arc_lengths = [0.6]
    if arc_radii is None:
        arc_radii = [np.inf,60, 30, 18.0, 12.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2]
    if lateral_offsets is None:
        lateral_offsets = [0.3, 0.5, 0.7]

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

    for lateral in lateral_offsets:
        t = np.linspace(0.0, 1.0, 12)
        x = np.zeros_like(t)
        z = np.zeros_like(t)
        for sign in (-1.0, 1.0):
            y = sign * lateral * t
            base_paths.append(np.stack([x, y, z], axis=1))
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
DEFAULT_DURATION_SEC = 20
DEFAULT_OUTPUT_FOLDER = str(DEFAULT_LOG_DIR)
DEFAULT_COLAB = False
DEFAULT_SUCCESS_THRESHOLD = 0.4
DEFAULT_DETOUR_RATIO = 1.15
DEFAULT_DRONE_COLLISION_RADIUS = 0.06
DEFAULT_DRONE_SIZE = 2.0 * DEFAULT_DRONE_COLLISION_RADIUS
DEFAULT_RIGID_EXTRA_MARGIN = 4.0 * DEFAULT_DRONE_SIZE
DEFAULT_GROUND_Z = 0.0
DEFAULT_MIN_GROUND_CLEARANCE = DEFAULT_DRONE_SIZE
DEFAULT_Z_MOTION_PENALTY_WEIGHT = 2.0
DEFAULT_EXECUTION_DISTANCE = 0.30
DEFAULT_FINAL_APPROACH_DISTANCE = 1.00

OBSTACLE_POINTS = None
OBSTACLE_KD_TREE = None


def _compute_endpoint_collision_mask(endpoints,
                                     kd_tree,
                                     formation_safe_radius):
    """Reject primitives whose endpoint circumcircle overlaps the obstacle point cloud."""
    if kd_tree is None:
        return np.zeros(endpoints.shape[0], dtype=bool)

    dists, _ = kd_tree.query(endpoints, k=1)
    return dists <= formation_safe_radius


def _compute_ground_collision_mask(endpoints,
                                   ground_z=DEFAULT_GROUND_Z,
                                   min_ground_clearance=DEFAULT_MIN_GROUND_CLEARANCE):
    """Reject primitives whose endpoint flies too close to the ground plane."""
    return endpoints[:, 2] <= (ground_z + min_ground_clearance)


def _compute_z_motion_penalties(rotated_paths, current_pos, penalty_weight=DEFAULT_Z_MOTION_PENALTY_WEIGHT):
    """Penalize any vertical motion to prefer planar detours over climbing or diving."""
    world_paths_z = rotated_paths[:, :, 2] + current_pos[2]
    max_vertical_motion = np.max(np.abs(world_paths_z - current_pos[2]), axis=1)
    return penalty_weight * max_vertical_motion


def _compute_alignment_direction(direction_to_target, current_vel, t_now):
    target_dir = direction_to_target / np.linalg.norm(direction_to_target)
    if t_now < 0.4:
        return target_dir

    v_norm = np.linalg.norm(current_vel)
    if v_norm > 1e-3:
        return current_vel / v_norm
    return target_dir


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
                        formation_safe_radius):
    """Direction-only local planner with obstacle and ground endpoint rejection."""
    direction_to_target = center_goal - current_pos
    distance_to_target = np.linalg.norm(direction_to_target)
    if distance_to_target < DEFAULT_SUCCESS_THRESHOLD:
        return center_goal
    if distance_to_target < DEFAULT_FINAL_APPROACH_DISTANCE:
        return current_pos + (
            min(DEFAULT_EXECUTION_DISTANCE, distance_to_target)
            * direction_to_target
            / distance_to_target
        )

    t_now = step * env.CTRL_TIMESTEP
    align_dir = _compute_alignment_direction(direction_to_target, current_vel, t_now)
    rot_matrix = _rotation_from_x_axis(align_dir)

    rotated_paths = paths @ rot_matrix.T
    endpoints = rotated_paths[:, -1, :] + current_pos
    goal_dists = np.linalg.norm(endpoints - center_goal, axis=1)
    z_motion_penalties = _compute_z_motion_penalties(rotated_paths, current_pos)
    obstacle_collision_mask = _compute_endpoint_collision_mask(endpoints,
                                                               OBSTACLE_KD_TREE,
                                                               formation_safe_radius)
    ground_collision_mask = _compute_ground_collision_mask(endpoints)
    collision_mask = obstacle_collision_mask | ground_collision_mask
    feasible_mask = ~collision_mask
    if not np.any(feasible_mask):
        return current_pos

    total_costs = np.where(feasible_mask, goal_dists + z_motion_penalties, np.inf)
    best_idx = np.argmin(total_costs)
    return endpoints[best_idx]


def _path_length(path_points):
    if len(path_points) < 2:
        return 0.0
    diffs = np.diff(np.array(path_points), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _pairwise_min_distance(positions):
    if len(positions) < 2:
        return np.inf

    min_dist = np.inf
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist < min_dist:
                min_dist = dist
    return min_dist


def _min_obstacle_clearance(positions, kd_tree, drone_radius):
    if kd_tree is None or len(positions) == 0:
        return np.inf
    dists, _ = kd_tree.query(np.asarray(positions), k=1)
    return float(np.min(dists) - drone_radius)


def _formation_circumradius(base_offsets):
    """Return the rigid formation circumradius around the leader reference point."""
    return float(np.max(np.linalg.norm(base_offsets, axis=1)))


def _rigid_collision_radius(base_offsets):
    """Use circumradius plus four drone sizes for conservative rigid-formation rejection."""
    return _formation_circumradius(base_offsets) + DEFAULT_RIGID_EXTRA_MARGIN


def _plot_goal_distance(goal_distance_history):
    if not goal_distance_history:
        return
    ts, ds = zip(*goal_distance_history)
    plt.figure()
    plt.plot(ts, ds, label="领导者到目标点距离")
    plt.xlabel("时间 [s]")
    plt.ylabel("距离 [m]")
    plt.title("刚性编队运动原语方法目标距离变化曲线")
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

    center_start = np.array([-3.0, 0.1, 1.5])
    center_goal = np.array([3.0, 0.1, 1.5])

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
    nominal_goals = center_goal + base_offsets
    output_path = resolve_output_path(output_folder)
    obstacle_specs = get_obstacle_box_specs()
    rigid_formation_radius = _rigid_collision_radius(base_offsets)

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
        output_folder=str(output_path),
    )

    time.sleep(1)
    pyb_client = env.getPyBulletClient()
    OBSTACLE_POINTS, OBSTACLE_KD_TREE = build_narrow_gap_kdtree(
        pyb_client,
        sample_volume=True,
    )

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
        output_folder=str(output_path),
        colab=colab,
    )
    controllers = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    action = np.zeros((num_drones, 4))
    start_wall_time = time.time()

    leader_positions = []
    goal_distance_history = []
    trajectory_histories = [[] for _ in range(num_drones)]
    min_pair_distance_history = []
    min_obstacle_clearance_history = []
    completion_time_sec = None

    last_obs = None
    for step in range(int(duration_sec * env.CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)
        last_obs = obs
        best_goals = np.array([obs[i][:3] for i in range(num_drones)])
        positions = np.array([obs[i][:3] for i in range(num_drones)])
        for drone_id in range(num_drones):
            trajectory_histories[drone_id].append(positions[drone_id].copy())
        min_pair_distance_history.append((step / env.CTRL_FREQ, _pairwise_min_distance(positions)))
        min_obstacle_clearance_history.append(
            (step / env.CTRL_FREQ, _min_obstacle_clearance(positions, OBSTACLE_KD_TREE, DEFAULT_DRONE_COLLISION_RADIUS))
        )

        leader_pos = obs[0][:3]
        leader_positions.append(leader_pos.copy())
        goal_distance_history.append((step / env.CTRL_FREQ, float(np.linalg.norm(center_goal - leader_pos))))
        if completion_time_sec is None and np.linalg.norm(center_goal - leader_pos) < DEFAULT_SUCCESS_THRESHOLD:
            completion_time_sec = step / env.CTRL_FREQ
            break

        for leader in leader_indices:
            current_pos = obs[leader][:3]
            current_vel = obs[leader][10:13]
            best_goal = _select_leader_goal(current_pos=current_pos,
                                            current_vel=current_vel,
                                            center_goal=center_goal,
                                            step=step,
                                            env=env,
                                            paths=path_library,
                                            formation_safe_radius=rigid_formation_radius)
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
    if last_obs is None:
        final_positions = init_xyzs.copy()
    else:
        final_positions = np.array([last_obs[i][:3] for i in range(num_drones)])
    final_goal_dists = np.linalg.norm(nominal_goals - final_positions, axis=1)
    min_pair_distance = (
        float(min(value for _, value in min_pair_distance_history))
        if min_pair_distance_history
        else np.inf
    )
    min_obstacle_clearance = (
        float(min(value for _, value in min_obstacle_clearance_history))
        if min_obstacle_clearance_history
        else np.inf
    )
    final_sim_time_sec = (
        completion_time_sec
        if completion_time_sec is not None
        else (float(min_pair_distance_history[-1][0]) if min_pair_distance_history else 0.0)
    )

    summary_lines = [
        "=== Fixed formation motion primitives summary ===",
        f"Reached goal (<{DEFAULT_SUCCESS_THRESHOLD:.2f} m): {reached}",
        (
            f"Completion time [s]: {completion_time_sec:.3f}"
            if completion_time_sec is not None
            else f"Completion time [s]: not reached (ran to {final_sim_time_sec:.3f})"
        ),
        f"Final leader distance to goal [m]: {final_dist:.3f}",
        f"Final goal distances [m]: {np.array2string(final_goal_dists, precision=3)}",
        f"Rigid formation collision radius [m]: {rigid_formation_radius:.3f}",
        f"Leader traveled length [m]: {traveled:.3f}",
        f"Straight-line length [m]: {straight:.3f}",
        f"Detour detected (>{DEFAULT_DETOUR_RATIO:.2f}x straight): {detour}",
        f"Minimum pairwise distance during run [m]: {min_pair_distance:.3f}",
        f"Minimum obstacle clearance during run [m]: {min_obstacle_clearance:.3f}",
    ]
    summary_path = save_summary_text(output_path, "fixed_formationMP_metrics.txt", summary_lines)
    saved_figure_paths = []

    if plot:
        safety_fig = plot_safety_histories(
            min_pair_distance_history,
            min_obstacle_clearance_history,
            title="刚性编队运动原语方法安全距离变化曲线",
        )
        traj_3d_fig = plot_3d_trajectories(
            trajectory_histories=trajectory_histories,
            nominal_goals=nominal_goals,
            final_positions=final_positions,
            obstacle_specs=obstacle_specs,
            title="刚性编队运动原语方法三维轨迹图",
        )
        top_view_fig = plot_top_view_trajectories(
            trajectory_histories=trajectory_histories,
            nominal_goals=nominal_goals,
            final_positions=final_positions,
            obstacle_specs=obstacle_specs,
            title="刚性编队运动原语方法俯视轨迹图",
        )
        figure_specs = [
            (safety_fig, output_path / "fixed_formationMP_safety.png"),
            (traj_3d_fig, output_path / "fixed_formationMP_3d.png"),
            (top_view_fig, output_path / "fixed_formationMP_top.png"),
        ]
        for fig, fig_path in figure_specs:
            if fig is not None:
                save_thesis_figure(fig, fig_path)
                saved_figure_paths.append(fig_path)
        plt.show()

    print()
    for line in summary_lines:
        print(line)
    print(f"Saved metrics text: {summary_path}")
    if saved_figure_paths:
        print("Saved figures:")
        for fig_path in saved_figure_paths:
            print(f"  {fig_path}")
    if record_video:
        if gui:
            print(f"Recorded video saved to: {output_path}")
        else:
            print(f"Recorded frames saved to: {output_path}")


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
