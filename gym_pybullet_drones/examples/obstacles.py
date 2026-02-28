"""Narrow-gap obstacle utilities for examples."""

import math
import numpy as np
import pybullet as p
from scipy.spatial import cKDTree

DEFAULT_GAP_CENTER = np.array([0.0, 0.0, 1.5])
DEFAULT_CAMERA_DISTANCE = 8.4
DEFAULT_GAP_OBSTACLE = dict(
    center_xy=(0.0, 0.0),
    center_z=1.5,
    total_length=4.0,
    total_width=16.0,
    total_height=3.0,
    gap_width=1.4,
    yaw_deg=90.0,
    resolution=0.1,
)


def _sample_box_surface(half_extents, resolution):
    """Sample points on box surface with a given resolution."""
    hx, hy, hz = half_extents
    xs = np.arange(-hx, hx + resolution * 0.5, resolution)
    ys = np.arange(-hy, hy + resolution * 0.5, resolution)
    zs = np.arange(-hz, hz + resolution * 0.5, resolution)

    faces = []

    yv, zv = np.meshgrid(ys, zs, indexing="xy")
    faces.append(np.column_stack([np.full(yv.size, hx), yv.ravel(), zv.ravel()]))
    faces.append(np.column_stack([np.full(yv.size, -hx), yv.ravel(), zv.ravel()]))

    xv, zv = np.meshgrid(xs, zs, indexing="xy")
    faces.append(np.column_stack([xv.ravel(), np.full(xv.size, hy), zv.ravel()]))
    faces.append(np.column_stack([xv.ravel(), np.full(xv.size, -hy), zv.ravel()]))

    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    faces.append(np.column_stack([xv.ravel(), yv.ravel(), np.full(xv.size, hz)]))
    faces.append(np.column_stack([xv.ravel(), yv.ravel(), np.full(xv.size, -hz)]))

    return np.unique(np.vstack(faces), axis=0)


def add_narrow_gap_obstacles(client_id, cfg=DEFAULT_GAP_OBSTACLE, rgba=(0.7, 0.2, 0.2, 0.9)):
    """Add two fixed blocks with a narrow gap and return obstacle surface points."""
    center_xy = cfg["center_xy"]
    center_z = cfg["center_z"]
    total_length = cfg["total_length"]
    total_width = cfg["total_width"]
    total_height = cfg["total_height"]
    gap_width = cfg["gap_width"]
    yaw_deg = cfg["yaw_deg"]
    resolution = cfg["resolution"]

    if gap_width >= total_width:
        raise ValueError("gap_width must be smaller than total_width to form two blocks.")

    wall_width = (total_width - gap_width) / 2.0
    if wall_width <= 0:
        raise ValueError("Computed wall width is non-positive; adjust total_width or gap_width.")

    half_extents = np.array([wall_width / 2.0, total_length / 2.0, total_height / 2.0])
    center_z = half_extents[2] if center_z is None else center_z

    yaw_rad = math.radians(yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    rot = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    quat = p.getQuaternionFromEuler([0.0, 0.0, yaw_rad])

    center = np.array([center_xy[0], center_xy[1], center_z])
    block_offsets = np.array(
        [
            [gap_width / 2.0 + half_extents[0], 0.0, 0.0],
            [-gap_width / 2.0 - half_extents[0], 0.0, 0.0],
        ]
    )

    surface_points = []
    local_surface = _sample_box_surface(half_extents, resolution)

    for offset in block_offsets:
        world_center = center + rot @ offset

        col_id = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents, physicsClientId=client_id
        )
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=rgba,
            physicsClientId=client_id,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=world_center,
            baseOrientation=quat,
            physicsClientId=client_id,
        )

        rotated_surface = (rot @ local_surface.T).T + world_center
        surface_points.append(rotated_surface)

    return np.vstack(surface_points)


def build_narrow_gap_kdtree(client_id, cfg=DEFAULT_GAP_OBSTACLE, rgba=(0.7, 0.2, 0.2, 0.9)):
    """Create narrow-gap obstacles and return (points, kd_tree)."""
    obstacle_points = add_narrow_gap_obstacles(client_id=client_id, cfg=cfg, rgba=rgba)
    obstacle_points = np.unique(obstacle_points, axis=0)
    kd_tree = cKDTree(obstacle_points) if obstacle_points.size > 0 else None
    return obstacle_points, kd_tree
