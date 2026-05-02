"""
Independent single-drone motion-primitive baseline for the narrow-gap experiment.

The environment, controller, and primitive library are aligned with the formation
planner, but each drone independently selects its own local primitive using only
the current observed states of obstacles and peers.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
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
from gym_pybullet_drones.examples.obstacles import (
    DEFAULT_CAMERA_DISTANCE,
    DEFAULT_GAP_CENTER,
    build_narrow_gap_kdtree,
)
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import str2bool, sync


def generate_directional_path_library(arc_lengths=None, arc_radii=None, delta_angle_deg=30):
    """Generate the same geometric primitive scales as the formation planner."""
    if arc_lengths is None:
        arc_lengths = [0.6]
        
    if arc_radii is None:
        arc_radii = [np.inf, 60.0, 30.0, 18.0, 12.0, 8.0, 6.0, 4.0,2.0, 1.0, 0.5, 0.25]
    if delta_angle_deg <= 0:
        raise ValueError("delta_angle_deg must be positive")

    num_samples = 12
    base_paths = []
    for radius in arc_radii:
        for length in arc_lengths:
            if math.isinf(radius):
                x = np.linspace(0.0, length, num_samples)
                y = np.zeros_like(x)
                z = np.zeros_like(x)
                base_paths.append(np.stack([x, y, z], axis=1))
                continue

            theta_max = length / radius
            thetas = np.linspace(0.0, theta_max, num_samples)
            x = radius * np.sin(thetas)
            y = radius * (1.0 - np.cos(thetas))
            z = np.zeros_like(thetas)
            arc = np.stack([x, y, z], axis=1)
            for angle in range(0, 360, delta_angle_deg):
                rot = R.from_euler("x", angle, degrees=True).as_matrix()
                base_paths.append((rot @ arc.T).T)

    return np.array(base_paths)


PATH_LIBRARY = generate_directional_path_library(delta_angle_deg=30)

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
DEFAULT_OUTPUT_FOLDER = str(DEFAULT_LOG_DIR)
DEFAULT_COLAB = False

DEFAULT_DRONE_SAFE_RADIUS = 0.06
DEFAULT_PAIR_SAFE_DISTANCE = 3.0 * DEFAULT_DRONE_SAFE_RADIUS
DEFAULT_INFLUENCE_DISTANCE = 0.4
DEFAULT_OBSTACLE_PENALTY_WEIGHT = 1.0
DEFAULT_INTER_DRONE_PENALTY_WEIGHT = 1.0
DEFAULT_EXPONENTIAL_PENALTY_GAIN = 4.0
DEFAULT_COLLISION_REJECTION_PENALTY = 1e6
DEFAULT_SUCCESS_THRESHOLD = 0.2
DEFAULT_HORIZON_TIME_SEC = 0.55
DEFAULT_EXECUTION_DISTANCE = 0.40

OBSTACLE_POINTS = None
OBSTACLE_KD_TREE = None


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


def _compute_goal_rewards(current_pos, endpoints, drone_goal):
    current_dist = np.linalg.norm(drone_goal - current_pos)
    end_dists = np.linalg.norm(endpoints - drone_goal, axis=1)
    return current_dist - end_dists


def _select_tracking_point(path_world, max_distance):
    """Track an early point on the chosen primitive to stay within controller authority."""
    if len(path_world) <= 1:
        return path_world[-1]

    diffs = np.diff(path_world, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    idx = int(np.searchsorted(cumulative, max_distance, side="right"))
    idx = min(max(1, idx), len(path_world) - 1)
    return path_world[idx]


def _compute_exponential_penalty(clearances, influence_distance, penalty_weight):
    if influence_distance <= 0.0:
        return np.zeros_like(clearances)

    normalized_gap = np.maximum(
        0.0,
        (influence_distance - clearances) / influence_distance,
    )
    return penalty_weight * (np.exp(DEFAULT_EXPONENTIAL_PENALTY_GAIN * normalized_gap) - 1.0)


def _compute_obstacle_penalties(
    rotated_paths,
    current_pos,
    kd_tree,
    drone_safe_radius,
    influence_distance,
    penalty_weight,
):
    if kd_tree is None:
        return (
            np.zeros(rotated_paths.shape[0]),
            np.zeros(rotated_paths.shape[0], dtype=bool),
        )

    path_world = rotated_paths + current_pos
    dists, _ = kd_tree.query(path_world.reshape(-1, 3), k=1)
    dists = dists.reshape(rotated_paths.shape[0], rotated_paths.shape[1])
    min_clearances = np.min(dists, axis=1) - drone_safe_radius
    endpoint_clearances = dists[:, -1] - drone_safe_radius
    penalties = _compute_exponential_penalty(
        clearances=min_clearances,
        influence_distance=influence_distance,
        penalty_weight=penalty_weight,
    )
    return penalties, endpoint_clearances <= 0.0


def _predict_other_trajectories(obs, ego_id, num_samples, horizon_time_sec):
    sample_alphas = np.linspace(0.0, 1.0, num_samples)
    trajectories = []
    endpoints = []
    for drone_id in range(len(obs)):
        if drone_id == ego_id:
            continue
        other_pos = np.array(obs[drone_id][:3])
        other_vel = np.array(obs[drone_id][10:13])
        pred = other_pos[None, :] + sample_alphas[:, None] * horizon_time_sec * other_vel[None, :]
        trajectories.append(pred)
        endpoints.append(pred[-1])
    return trajectories, endpoints


def _compute_inter_drone_penalties(
    rotated_paths,
    current_pos,
    other_trajectories,
    other_endpoints,
    pair_safe_distance,
    influence_distance,
    penalty_weight,
):
    if not other_endpoints:
        return (
            np.zeros(rotated_paths.shape[0]),
            np.zeros(rotated_paths.shape[0], dtype=bool),
        )

    path_world = rotated_paths + current_pos
    endpoint_positions = path_world[:, -1, :]
    min_dists = np.full(rotated_paths.shape[0], np.inf)
    endpoint_min_dists = np.full(rotated_paths.shape[0], np.inf)
    for other_traj, other_endpoint in zip(other_trajectories, other_endpoints):
        dists = np.linalg.norm(path_world - other_traj[None, :, :], axis=2)
        min_dists = np.minimum(min_dists, np.min(dists, axis=1))
        endpoint_dists = np.linalg.norm(endpoint_positions - other_endpoint[None, :], axis=1)
        endpoint_min_dists = np.minimum(endpoint_min_dists, endpoint_dists)

    penalties = _compute_exponential_penalty(
        clearances=min_dists - pair_safe_distance,
        influence_distance=influence_distance,
        penalty_weight=penalty_weight,
    )
    collision_mask = (endpoint_min_dists - pair_safe_distance) <= 0.0
    return penalties, collision_mask


def _select_local_goal(
    current_pos,
    current_vel,
    drone_goal,
    step,
    env,
    paths,
    other_trajectories,
    other_endpoints,
):
    direction_to_target = drone_goal - current_pos
    distance_to_target = np.linalg.norm(direction_to_target)
    if distance_to_target < DEFAULT_SUCCESS_THRESHOLD:
        return drone_goal, -1, {
            "goal_reward": 0.0,
            "feasible_count": 0,
            "score": 0.0,
        }

    t_now = step * env.CTRL_TIMESTEP
    align_dir = _compute_alignment_direction(direction_to_target, current_vel, t_now)
    rot_matrix = _rotation_from_x_axis(align_dir)

    rotated_paths = paths @ rot_matrix.T
    endpoints = rotated_paths[:, -1, :] + current_pos

    goal_rewards = _compute_goal_rewards(current_pos, endpoints, drone_goal)
    obstacle_penalties, obstacle_collision_mask = _compute_obstacle_penalties(
        rotated_paths=rotated_paths,
        current_pos=current_pos,
        kd_tree=OBSTACLE_KD_TREE,
        drone_safe_radius=DEFAULT_DRONE_SAFE_RADIUS,
        influence_distance=DEFAULT_INFLUENCE_DISTANCE,
        penalty_weight=DEFAULT_OBSTACLE_PENALTY_WEIGHT,
    )
    inter_drone_penalties, inter_drone_collision_mask = _compute_inter_drone_penalties(
        rotated_paths=rotated_paths,
        current_pos=current_pos,
        other_trajectories=other_trajectories,
        other_endpoints=other_endpoints,
        pair_safe_distance=DEFAULT_PAIR_SAFE_DISTANCE,
        influence_distance=DEFAULT_INFLUENCE_DISTANCE,
        penalty_weight=DEFAULT_INTER_DRONE_PENALTY_WEIGHT,
    )
    collision_mask = obstacle_collision_mask | inter_drone_collision_mask
    feasible_mask = ~collision_mask
    scores = goal_rewards - obstacle_penalties - inter_drone_penalties
    scores[collision_mask] = -DEFAULT_COLLISION_REJECTION_PENALTY
    if not np.any(feasible_mask):
        return current_pos.copy(), -1, {
            "goal_reward": 0.0,
            "obstacle_penalty": 0.0,
            "inter_drone_penalty": 0.0,
            "feasible_count": 0,
            "score": -DEFAULT_COLLISION_REJECTION_PENALTY,
        }
    best_idx = int(np.argmax(scores))
    best_path_world = rotated_paths[best_idx] + current_pos
    tracking_goal = _select_tracking_point(best_path_world, DEFAULT_EXECUTION_DISTANCE)
    return tracking_goal, best_idx, {
        "goal_reward": float(goal_rewards[best_idx]),
        "obstacle_penalty": float(obstacle_penalties[best_idx]),
        "inter_drone_penalty": float(inter_drone_penalties[best_idx]),
        "feasible_count": int(np.count_nonzero(feasible_mask)),
        "score": float(scores[best_idx]),
    }


def _path_length(path_points):
    if len(path_points) < 2:
        return 0.0
    diffs = np.diff(np.array(path_points), axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _pairwise_min_distance(positions):
    positions = np.asarray(positions)
    if len(positions) < 2:
        return np.inf

    min_dist = np.inf
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = float(np.linalg.norm(positions[i] - positions[j]))
            if dist < min_dist:
                min_dist = dist
    return min_dist


def _min_obstacle_clearance(positions, kd_tree):
    if kd_tree is None or len(positions) == 0:
        return np.inf
    dists, _ = kd_tree.query(np.asarray(positions), k=1)
    return float(np.min(dists) - DEFAULT_DRONE_SAFE_RADIUS)


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
    """Run the independent single-drone motion-primitive baseline."""
    global OBSTACLE_POINTS, OBSTACLE_KD_TREE

    center_start = np.array([-3.0, 0.0, 1.5])
    center_goal = np.array([3.0, 0.0, 1.5])

    triangle_offset = 0.8
    sqrt3_over_2 = np.sqrt(3) / 2.0
    base_offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [triangle_offset, 0.0, 0.0],
            [-triangle_offset / 2.0, sqrt3_over_2 * triangle_offset, 0.0],
            [-triangle_offset / 2.0, -sqrt3_over_2 * triangle_offset, 0.0],
        ]
    )

    init_xyzs = center_start + base_offsets
    init_rpys = np.zeros((num_drones, 3))
    nominal_goals = center_goal + base_offsets
    output_path = resolve_output_path(output_folder)
    obstacle_specs = get_obstacle_box_specs()

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

    trajectory_histories = [[] for _ in range(num_drones)]
    goal_distance_histories = [[] for _ in range(num_drones)]
    planning_times_ms = []
    per_drone_planning_times_ms = [[] for _ in range(num_drones)]
    min_pair_distance_history = []
    min_obstacle_clearance_history = []
    selected_primitive_histories = [[] for _ in range(num_drones)]
    last_obs = None
    termination_time_sec = None
    for step in range(int(duration_sec * env.CTRL_FREQ)):
        obs, _, _, _, _ = env.step(action)
        last_obs = obs
        best_goals = np.array([obs[i][:3] for i in range(num_drones)])

        positions = np.array([obs[i][:3] for i in range(num_drones)])
        current_goal_dists = np.linalg.norm(nominal_goals - positions, axis=1)
        min_pair_distance_history.append(
            (step / env.CTRL_FREQ, _pairwise_min_distance(positions))
        )
        min_obstacle_clearance_history.append(
            (step / env.CTRL_FREQ, _min_obstacle_clearance(positions, OBSTACLE_KD_TREE))
        )

        for drone_id in range(num_drones):
            current_pos = np.array(obs[drone_id][:3])
            trajectory_histories[drone_id].append(current_pos.copy())
            goal_distance_histories[drone_id].append(
                (
                    step / env.CTRL_FREQ,
                    float(current_goal_dists[drone_id]),
                )
            )

        if np.all(current_goal_dists < DEFAULT_SUCCESS_THRESHOLD):
            termination_time_sec = step / env.CTRL_FREQ
            break

        cycle_plan_start = time.perf_counter()
        for drone_id in range(num_drones):
            current_pos = np.array(obs[drone_id][:3])
            current_vel = np.array(obs[drone_id][10:13])

            other_trajectories, other_endpoints = _predict_other_trajectories(
                obs=obs,
                ego_id=drone_id,
                num_samples=PATH_LIBRARY.shape[1],
                horizon_time_sec=DEFAULT_HORIZON_TIME_SEC,
            )

            drone_plan_start = time.perf_counter()
            best_goal, best_idx, _ = _select_local_goal(
                current_pos=current_pos,
                current_vel=current_vel,
                drone_goal=nominal_goals[drone_id],
                step=step,
                env=env,
                paths=PATH_LIBRARY,
                other_trajectories=other_trajectories,
                other_endpoints=other_endpoints,
            )
            per_drone_planning_times_ms[drone_id].append(
                1000.0 * (time.perf_counter() - drone_plan_start)
            )
            selected_primitive_histories[drone_id].append(best_idx)
            best_goals[drone_id] = best_goal

        planning_times_ms.append(1000.0 * (time.perf_counter() - cycle_plan_start))

        for drone_id in range(num_drones):
            action[drone_id, :], _, _ = controllers[drone_id].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[drone_id],
                target_pos=best_goals[drone_id],
                target_rpy=init_rpys[drone_id, :],
            )
            logger.log(
                drone=drone_id,
                timestamp=step / env.CTRL_FREQ,
                state=obs[drone_id],
                control=np.hstack([best_goals[drone_id], init_rpys[drone_id, :], np.zeros(6)]),
            )

        env.render()
        if gui:
            sync(step, start_wall_time, env.CTRL_TIMESTEP)

    env.close()

    if last_obs is None:
        final_positions = init_xyzs.copy()
    else:
        final_positions = np.array([last_obs[i][:3] for i in range(num_drones)])

    final_goal_dists = np.linalg.norm(nominal_goals - final_positions, axis=1)
    reached = bool(np.all(final_goal_dists < DEFAULT_SUCCESS_THRESHOLD))
    path_lengths = [_path_length(path) for path in trajectory_histories]
    mean_cycle_plan_ms = float(np.mean(planning_times_ms)) if planning_times_ms else 0.0
    p95_cycle_plan_ms = float(np.percentile(planning_times_ms, 95)) if planning_times_ms else 0.0
    max_cycle_plan_ms = float(np.max(planning_times_ms)) if planning_times_ms else 0.0
    mean_per_drone_plan_ms = [
        float(np.mean(times)) if times else 0.0 for times in per_drone_planning_times_ms
    ]
    unique_primitives_per_drone = [
        len({idx for idx in history if idx >= 0}) for history in selected_primitive_histories
    ]
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
    completion_time_sec = termination_time_sec if termination_time_sec is not None else None
    if min_pair_distance_history:
        last_timestamp = float(min_pair_distance_history[-1][0])
    else:
        last_timestamp = 0.0
    final_sim_time_sec = completion_time_sec if completion_time_sec is not None else last_timestamp

    summary_lines = [
        "=== Independent single-drone motion primitives summary ===",
        f"Primitive library size: {PATH_LIBRARY.shape[0]}",
        f"Reached all assigned goals (<{DEFAULT_SUCCESS_THRESHOLD:.2f} m): {reached}",
        (
            f"Completion time [s]: {completion_time_sec:.3f}"
            if completion_time_sec is not None
            else f"Completion time [s]: not reached (ran to {final_sim_time_sec:.3f})"
        ),
        f"Final goal distances [m]: {np.array2string(final_goal_dists, precision=3)}",
        f"Path lengths [m]: {np.array2string(np.array(path_lengths), precision=3)}",
        f"Mean cycle planning time: {mean_cycle_plan_ms:.3f} ms",
        f"95th-percentile cycle planning time: {p95_cycle_plan_ms:.3f} ms",
        f"Max cycle planning time: {max_cycle_plan_ms:.3f} ms",
        "Mean per-drone planning times [ms]: "
        f"{np.array2string(np.array(mean_per_drone_plan_ms), precision=3)}",
        "Unique primitive indices selected per drone: "
        f"{np.array2string(np.array(unique_primitives_per_drone), precision=0)}",
        f"Minimum pairwise distance during run: {min_pair_distance:.3f} m",
        f"Minimum obstacle clearance during run: {min_obstacle_clearance:.3f} m",
    ]
    summary_path = save_summary_text(output_path, "mp_independent_metrics.txt", summary_lines)

    saved_figure_paths = []

    if plot:
        safety_fig = plot_safety_histories(
            min_pair_distance_history,
            min_obstacle_clearance_history,
            title="分布式运动原语方法安全距离变化曲线",
        )
        traj_3d_fig = plot_3d_trajectories(
            trajectory_histories=trajectory_histories,
            nominal_goals=nominal_goals,
            final_positions=final_positions,
            obstacle_specs=obstacle_specs,
            title="分布式运动原语方法三维轨迹图",
        )
        top_view_fig = plot_top_view_trajectories(
            trajectory_histories=trajectory_histories,
            nominal_goals=nominal_goals,
            final_positions=final_positions,
            obstacle_specs=obstacle_specs,
            title="分布式运动原语方法俯视轨迹图",
        )
        figure_specs = [
            (safety_fig, output_path / "mp_independent_safety.png"),
            (traj_3d_fig, output_path / "mp_independent_3d.png"),
            (top_view_fig, output_path / "mp_independent_top.png"),
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
        description="Four-drone narrow-gap baseline: independent motion primitives for each drone"
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
