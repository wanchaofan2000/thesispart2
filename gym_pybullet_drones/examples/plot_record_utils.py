"""Shared plotting and record helpers for narrow-gap experiment scripts."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from gym_pybullet_drones.examples.obstacles import DEFAULT_GAP_OBSTACLE

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = PACKAGE_ROOT / "log"
THESIS_EXPORT_DPI = 300
THESIS_FONT_SIZE = 10.5
THESIS_FONT_FILES = [
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"),
]
THESIS_SERIF_FONTS = [
    "Noto Serif CJK SC",
    "Noto Serif CJK JP",
    "SimSun",
    "Songti SC",
    "STSong",
    "Source Han Serif SC",
    "Nimbus Roman",
    "Times New Roman",
    "DejaVu Serif",
]


def _register_thesis_fonts():
    for font_path in THESIS_FONT_FILES:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
            except RuntimeError:
                continue


def _resolve_available_thesis_fonts():
    _register_thesis_fonts()
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    resolved = [font_name for font_name in THESIS_SERIF_FONTS if font_name in available_fonts]
    return resolved or ["DejaVu Serif"]


def configure_thesis_plot_style():
    available_fonts = _resolve_available_thesis_fonts()
    plt.rcParams.update(
        {
            "font.family": available_fonts,
            "font.serif": available_fonts,
            "axes.unicode_minus": False,
            "font.size": THESIS_FONT_SIZE,
            "axes.labelsize": THESIS_FONT_SIZE,
            "axes.titlesize": THESIS_FONT_SIZE,
            "xtick.labelsize": THESIS_FONT_SIZE,
            "ytick.labelsize": THESIS_FONT_SIZE,
            "legend.fontsize": THESIS_FONT_SIZE,
        }
    )


configure_thesis_plot_style()


def resolve_output_path(output_folder):
    output_path = Path(output_folder).expanduser()
    if not output_path.is_absolute():
        output_path = DEFAULT_LOG_DIR / output_path
    output_path = output_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_summary_text(output_path, filename, summary_lines):
    summary_path = output_path / filename
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return summary_path


def save_thesis_figure(fig, fig_path):
    fig.savefig(fig_path, dpi=THESIS_EXPORT_DPI, bbox_inches="tight", facecolor="white")
    return fig_path


def _lightened_rgba(base_color, blend_ratio, alpha):
    base_rgb = np.array(to_rgb(base_color))
    light_rgb = (1.0 - blend_ratio) * np.ones(3) + blend_ratio * base_rgb
    return (*light_rgb, alpha)


def plot_safety_histories(
    min_pair_distance_history,
    min_obstacle_clearance_history,
    title="安全距离变化曲线",
):
    if not min_pair_distance_history and not min_obstacle_clearance_history:
        return None

    fig = plt.figure()
    if min_pair_distance_history:
        ts, ds = zip(*min_pair_distance_history)
        plt.plot(ts, ds, label="最小机间距离")
    if min_obstacle_clearance_history:
        ts, ds = zip(*min_obstacle_clearance_history)
        plt.plot(ts, ds, label="最小障碍净距")
    plt.xlabel("时间 [s]")
    plt.ylabel("距离 [m]")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    return fig


def get_obstacle_box_specs(cfg=DEFAULT_GAP_OBSTACLE):
    center_xy = np.asarray(cfg["center_xy"], dtype=float)
    center_z = float(cfg["center_z"])
    total_length = float(cfg["total_length"])
    total_width = float(cfg["total_width"])
    total_height = float(cfg["total_height"])
    gap_width = float(cfg["gap_width"])
    yaw_rad = math.radians(float(cfg["yaw_deg"]))

    wall_width = (total_width - gap_width) / 2.0
    half_extents = np.array([wall_width / 2.0, total_length / 2.0, total_height / 2.0], dtype=float)
    rot = np.array(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    center = np.array([center_xy[0], center_xy[1], center_z], dtype=float)
    block_offsets = np.array(
        [
            [gap_width / 2.0 + half_extents[0], 0.0, 0.0],
            [-gap_width / 2.0 - half_extents[0], 0.0, 0.0],
        ],
        dtype=float,
    )

    specs = []
    for offset in block_offsets:
        specs.append(
            {
                "center": center + rot @ offset,
                "half_extents": half_extents.copy(),
                "rotation": rot.copy(),
            }
        )
    return specs


def _box_vertices(center, half_extents, rotation):
    hx, hy, hz = half_extents
    local_vertices = np.array(
        [
            [hx, hy, hz],
            [hx, hy, -hz],
            [hx, -hy, hz],
            [hx, -hy, -hz],
            [-hx, hy, hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [-hx, -hy, -hz],
        ],
        dtype=float,
    )
    return (rotation @ local_vertices.T).T + center


def _plot_obstacle_boxes_3d(ax, obstacle_specs, face_color="#c96f5d", alpha=0.10):
    face_indices = [
        [0, 2, 6, 4],
        [1, 3, 7, 5],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 1, 3, 2],
        [4, 5, 7, 6],
    ]
    all_vertices = []
    for i, spec in enumerate(obstacle_specs):
        vertices = _box_vertices(spec["center"], spec["half_extents"], spec["rotation"])
        faces = [[vertices[idx] for idx in face] for face in face_indices]
        poly = Poly3DCollection(
            faces,
            facecolors=face_color,
            edgecolors=_lightened_rgba(face_color, 0.85, 0.20),
            linewidths=0.5,
            alpha=alpha,
        )
        if i == 0:
            poly.set_label("障碍物")
        ax.add_collection3d(poly)
        all_vertices.append(vertices)
    return np.vstack(all_vertices) if all_vertices else np.empty((0, 3))


def _plot_obstacle_boxes_top(ax, obstacle_specs, face_color="#c96f5d", alpha=0.14):
    all_xy = []
    for i, spec in enumerate(obstacle_specs):
        vertices = _box_vertices(spec["center"], spec["half_extents"], spec["rotation"])
        top_polygon = vertices[[0, 2, 6, 4], :2]
        patch = Polygon(
            top_polygon,
            closed=True,
            facecolor=face_color,
            edgecolor=_lightened_rgba(face_color, 0.85, 0.25),
            linewidth=0.8,
            alpha=alpha,
            label="障碍物" if i == 0 else None,
        )
        ax.add_patch(patch)
        all_xy.append(top_polygon)
    return np.vstack(all_xy) if all_xy else np.empty((0, 2))


def _set_equal_3d_axes(ax, points):
    if points.size == 0:
        return

    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius < 1e-6:
        radius = 1.0

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_3d_trajectories(
    trajectory_histories,
    nominal_goals,
    final_positions,
    obstacle_specs=None,
    title="三维轨迹图",
):
    if not any(history for history in trajectory_histories):
        return None

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    all_points = [nominal_goals, final_positions]
    cmap = plt.get_cmap("tab10")

    if obstacle_specs:
        obstacle_vertices = _plot_obstacle_boxes_3d(ax, obstacle_specs)
        if obstacle_vertices.size > 0:
            all_points.append(obstacle_vertices)

    for drone_id, history in enumerate(trajectory_histories):
        if not history:
            continue
        path = np.asarray(history)
        all_points.append(path)
        base_color = cmap(drone_id % cmap.N)

        if len(path) >= 2:
            segments = np.stack([path[:-1], path[1:]], axis=1)
            blend_ratios = np.linspace(0.20, 1.00, len(segments))
            alphas = np.linspace(0.35, 0.95, len(segments))
            segment_colors = [
                _lightened_rgba(base_color, blend, alpha)
                for blend, alpha in zip(blend_ratios, alphas)
            ]
            line_collection = Line3DCollection(
                segments,
                colors=segment_colors,
                linewidths=2.0,
            )
            ax.add_collection3d(line_collection)
        else:
            ax.scatter(path[0, 0], path[0, 1], path[0, 2], s=16, color=base_color)

        ax.plot([], [], [], color=base_color, linewidth=2.0, label=f"无人机{drone_id}")
        ax.scatter(
            final_positions[drone_id, 0],
            final_positions[drone_id, 1],
            final_positions[drone_id, 2],
            s=36,
            marker="o",
            color=base_color,
        )
        ax.scatter(
            nominal_goals[drone_id, 0],
            nominal_goals[drone_id, 1],
            nominal_goals[drone_id, 2],
            s=140,
            marker="*",
            color=base_color,
            alpha=0.35,
        )
        ax.text(
            nominal_goals[drone_id, 0] + 0.05,
            nominal_goals[drone_id, 1] + 0.05,
            nominal_goals[drone_id, 2] + 0.05,
            f"G{drone_id}",
            fontsize=9,
        )

    stacked_points = np.vstack(all_points)
    _set_equal_3d_axes(ax, stacked_points)
    ax.set_xlabel("x 坐标 [m]")
    ax.set_ylabel("y 坐标 [m]")
    ax.set_zlabel("z 坐标 [m]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    return fig


def plot_top_view_trajectories(
    trajectory_histories,
    nominal_goals,
    final_positions,
    obstacle_specs=None,
    title="俯视轨迹图",
):
    if not any(history for history in trajectory_histories):
        return None

    fig, ax = plt.subplots()
    all_xy = [nominal_goals[:, :2], final_positions[:, :2]]
    cmap = plt.get_cmap("tab10")

    if obstacle_specs:
        obstacle_xy = _plot_obstacle_boxes_top(ax, obstacle_specs)
        if obstacle_xy.size > 0:
            all_xy.append(obstacle_xy)

    for drone_id, history in enumerate(trajectory_histories):
        if not history:
            continue
        path = np.asarray(history)
        all_xy.append(path[:, :2])
        base_color = cmap(drone_id % cmap.N)

        if len(path) >= 2:
            segments = np.stack([path[:-1, :2], path[1:, :2]], axis=1)
            blend_ratios = np.linspace(0.20, 1.00, len(segments))
            alphas = np.linspace(0.35, 0.95, len(segments))
            segment_colors = [
                _lightened_rgba(base_color, blend, alpha)
                for blend, alpha in zip(blend_ratios, alphas)
            ]
            line_collection = LineCollection(
                segments,
                colors=segment_colors,
                linewidths=2.0,
            )
            ax.add_collection(line_collection)
        else:
            ax.scatter(path[0, 0], path[0, 1], s=16, color=base_color)

        ax.plot([], [], color=base_color, linewidth=2.0, label=f"无人机{drone_id}")
        ax.scatter(
            final_positions[drone_id, 0],
            final_positions[drone_id, 1],
            s=36,
            marker="o",
            color=base_color,
        )
        ax.scatter(
            nominal_goals[drone_id, 0],
            nominal_goals[drone_id, 1],
            s=140,
            marker="*",
            color=base_color,
            alpha=0.35,
        )
        ax.text(
            nominal_goals[drone_id, 0] + 0.05,
            nominal_goals[drone_id, 1] + 0.05,
            f"G{drone_id}",
            fontsize=9,
        )

    stacked_xy = np.vstack(all_xy)
    mins = np.min(stacked_xy, axis=0)
    maxs = np.max(stacked_xy, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    if radius < 1e-6:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_xlabel("x 坐标 [m]")
    ax.set_ylabel("y 坐标 [m]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    return fig
