"""
STARK VISION — 3D Holographic Perspective Geometry Engine.

Renders true 3D wireframe models (rotating Arc Reactor gyroscope core,
volumetric target brackets, and gimbal rings) projected onto 2D camera frames
with sub-pixel anti-aliasing and depth parallax.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

# BGR Design Tokens
ARC_CYAN = (255, 240, 0)
PLASMA_BLUE = (255, 144, 30)
STARK_GOLD = (0, 191, 255)
HUD_WHITE = (245, 245, 245)


def rotate_x(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate (N, 3) points around the X-axis."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_a, -sin_a], [0.0, sin_a, cos_a]],
        dtype=np.float32,
    )
    return points @ rot.T


def rotate_y(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate (N, 3) points around the Y-axis."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array(
        [[cos_a, 0.0, sin_a], [0.0, 1.0, 0.0], [-sin_a, 0.0, cos_a]],
        dtype=np.float32,
    )
    return points @ rot.T


def rotate_z(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate (N, 3) points around the Z-axis."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    rot = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return points @ rot.T


def project_points(
    points_3d: np.ndarray,
    center: tuple[int, int],
    focal_length: float = 140.0,
    camera_dist: float = 90.0,
) -> np.ndarray:
    """
    Project 3D points (N, 3) into 2D pixel coordinates with perspective division.

    Returns:
        (N, 2) integer pixel array.
    """
    cx, cy = center
    z = points_3d[:, 2] + camera_dist
    z = np.maximum(z, 1.0)  # Avoid division by zero
    scale = focal_length / z

    u = cx + points_3d[:, 0] * scale
    v = cy + points_3d[:, 1] * scale
    return np.column_stack((u, v)).astype(np.int32)


def generate_ring_vertices(radius: float, num_pts: int = 24) -> np.ndarray:
    """Generate 3D vertices for a circle in the XY plane."""
    theta = np.linspace(0, 2 * math.pi, num_pts, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)
    return np.column_stack((x, y, z)).astype(np.float32)


def generate_octahedron(radius: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate 3D vertices and wireframe edges for an octahedron gem."""
    vertices = np.array(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 2),
        (2, 1),
        (1, 3),
        (3, 0),
        (0, 4),
        (1, 4),
        (2, 4),
        (3, 4),
        (0, 5),
        (1, 5),
        (2, 5),
        (3, 5),
    ]
    return vertices, edges


def draw_3d_arc_reactor(
    frame: np.ndarray,
    center: tuple[int, int],
    frame_count: int,
    base_radius: float = 24.0,
) -> None:
    """
    Render a true 3D rotating holographic Arc Reactor gyroscope with depth parallax.
    """
    t = frame_count * 0.04

    # 1. Outer Ring: Tilted around X by 30 degrees, rotating around Z
    outer_pts = generate_ring_vertices(base_radius, num_pts=20)
    outer_pts = rotate_x(outer_pts, math.radians(28))
    outer_pts = rotate_z(outer_pts, t * 1.2)
    p2d_outer = project_points(outer_pts, center)

    # Draw outer segmented arcs with glow
    for i in range(len(p2d_outer)):
        if (i % 5) != 4:  # 4 segmented gaps
            p1 = tuple(p2d_outer[i])
            p2 = tuple(p2d_outer[(i + 1) % len(p2d_outer)])
            cv2.line(frame, p1, p2, PLASMA_BLUE, 2, cv2.LINE_AA)
            cv2.line(frame, p1, p2, STARK_GOLD, 1, cv2.LINE_AA)

    # 2. Middle Ring: Tilted around Y by -25 degrees, counter-rotating
    mid_pts = generate_ring_vertices(base_radius * 0.72, num_pts=16)
    mid_pts = rotate_y(mid_pts, math.radians(-25))
    mid_pts = rotate_z(mid_pts, -t * 1.5)
    p2d_mid = project_points(mid_pts, center)

    for i in range(len(p2d_mid)):
        if (i % 4) != 3:
            p1 = tuple(p2d_mid[i])
            p2 = tuple(p2d_mid[(i + 1) % len(p2d_mid)])
            cv2.line(frame, p1, p2, ARC_CYAN, 1, cv2.LINE_AA)

    # 3. Inner 3D Octahedron Core Gem rotating on 3 axes
    core_pts, edges = generate_octahedron(radius=base_radius * 0.38)
    core_pts = rotate_x(core_pts, t * 0.9)
    core_pts = rotate_y(core_pts, t * 1.1)
    core_pts = rotate_z(core_pts, t * 0.7)
    p2d_core = project_points(core_pts, center)

    for idx1, idx2 in edges:
        p1 = tuple(p2d_core[idx1])
        p2 = tuple(p2d_core[idx2])
        cv2.line(frame, p1, p2, ARC_CYAN, 1, cv2.LINE_AA)

    # Center glowing energy pip
    cv2.circle(frame, center, 3, HUD_WHITE, -1, cv2.LINE_AA)
    cv2.circle(frame, center, 5, ARC_CYAN, 1, cv2.LINE_AA)
