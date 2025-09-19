# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import typing
import numpy as np
from math import floor
from bl_math import lerp
from mathutils import Vector


def all_equal(sequence, key: typing.Callable | None = None):
    if key is None:
        sequence_iter = iter(sequence)
        first = next(sequence_iter)
        for i in sequence_iter:
            if i != first:
                return False
    else:
        sequence_iter = iter(sequence)
        first = key(next(sequence_iter))
        for i in sequence_iter:
            if key(i) != first:
                return False
    return True


def vec_isclose(a, b, abs_tol: float = 0.00001):
    return all(math.isclose(a1, b1, abs_tol=abs_tol) for a1, b1 in zip(a, b))


def vec_isclose_to_uniform(delta: Vector, abs_tol: float = 0.00001):
    return all(math.isclose(component, 1.0, abs_tol=abs_tol) for component in delta)


def vec_isclose_to_zero(delta: Vector, abs_tol: float = 0.00001):
    return all(math.isclose(component, 0.0, abs_tol=abs_tol) for component in delta)


def safe_div(a, b):
    return a / b if b != 0 else 0

# Source: https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56


def inv_lerp(a: float, b: float, v: float) -> float:  # ratio
    ratio_range = b - a
    return ((v - a) / ratio_range) if ratio_range else 0


def remap(i_min: float, i_max: float, o_min: float, o_max: float, v: float) -> float:
    return lerp(o_min, o_max, inv_lerp(i_min, i_max, v))


def weighted_linear_space(start, stop, w):
    if len(w) == 1:
        return np.array([start, stop], dtype=np.float32)

    start = np.array([start], dtype=np.float32)
    stop = np.array([stop], dtype=np.float32)

    if not isinstance(w, np.ndarray):
        w = np.array(w, dtype=np.float32)

    nw = w / np.sum(w)  # Normalize weights
    cum_w = np.insert(np.cumsum(nw), 0, 0)  # Create cumulative weight sum
    return start + np.outer(cum_w, stop - start)


def vec_to_cardinal(v):
    x, y = v
    if abs(x) >= abs(y):
        return Vector([np.sign(x), 0])
    else:
        return Vector([0, np.sign(y)])


def round_threshold(a, min_clip):
    return round(float(a) / min_clip) * min_clip


def fract(a: float) -> float:
    return a - floor(a)


def wrap(value, minimum=0, maximum=1):
    wrap_range = maximum - minimum
    return value - (wrap_range * floor((value - minimum) / wrap_range)) if (wrap_range != 0) else minimum


def wrap_line(start, width, min_bound, max_bound, default=None):
    try:
        return (start - min_bound) % (max_bound - min_bound - width) + min_bound
    except ZeroDivisionError:
        if default is None:
            raise ZeroDivisionError
        else:
            return default


def power_of_2_round(val: int) -> int:
    if not val:
        return 0
    return 2 ** round(math.log2(abs(val)))


def power_of_2_ceil(val: int) -> int:
    if not val:
        return 0
    return 2 ** math.ceil(math.log2(abs(val)))


def power_of_2_floor(val: int) -> int:
    if not val:
        return 0
    return 2 ** floor(math.log2(abs(val)))


def is_power_of_2(n: int) -> bool:
    assert (not n <= 0), 'Value error'
    return (n & (n - 1)) == 0


def closest_pt_to_line(pt: Vector, l_a: Vector, l_b: Vector):
    line_vec = l_b - l_a
    pt_vec = pt - l_a
    if not (line_len_squared := line_vec.dot(line_vec)):
        return l_a

    projection = pt_vec.dot(line_vec) / line_len_squared
    if projection < 0:
        return l_a
    elif projection > 1:
        return l_b
    return l_a + projection * line_vec


def find_closest_edge_3d_to_2d(mouse_pos, face, umesh, region, rv3d):
    pt = Vector(mouse_pos)
    mat = umesh.obj.matrix_world
    min_edge = None
    min_dist = float('inf')
    for e in face.edges:
        v_a, v_b = e.verts

        co_a = loc3d_to_reg2d_safe(region, rv3d, mat @ v_a.co)
        co_b = loc3d_to_reg2d_safe(region, rv3d, mat @ v_b.co)

        close_pt = closest_pt_to_line(pt, co_a, co_b)
        dist = (close_pt - pt).length
        if dist < min_dist:
            min_edge = e
            min_dist = dist

    return min_edge, min_dist


def loc3d_to_reg2d_safe(region, rv3d, coord, push_forward=0.01):
    prj = rv3d.perspective_matrix @ Vector((*coord, 1.0))

    for i in range(2, 12):
        if prj.w <= 0.0:
            view_dir = (rv3d.view_rotation @ Vector((0.0, 0.0, -1.0))).normalized()
            coord = coord + view_dir * push_forward
            prj = rv3d.perspective_matrix @ Vector((*coord, 1.0))
            push_forward *= i
        else:
            break

    width_half = region.width / 2.0
    height_half = region.height / 2.0

    return Vector((
        width_half + width_half * (prj.x / prj.w),
        height_half + height_half * (prj.y / prj.w),
    ))


def np_vec_dot(a, b):
    return np.einsum('ij,ij->i', a, b)


def np_vec_normalized(a, keepdims=True):
    return np.linalg.norm(a, axis=1, keepdims=keepdims)

# def closest_pts_to_lines(pt: np.ndarray, l_a: np.ndarray, l_b: np.ndarray) -> np.ndarray:
#     line_vecs = l_b - l_a
#     pt_vecs = pt - l_a
#
#     line_len_squared = np.sum(line_vecs ** 2, axis=1)
#     projections = np.sum(pt_vecs * line_vecs, axis=1) / line_len_squared
#
#     # Restrict projections in the range [0, 1] for those that are inside segments
#     projections_clipped = np.clip(projections, 0, 1)
#
#     closest_pts = l_a + projections_clipped[:, np.newaxis] * line_vecs
#
#     return closest_pts

# def test(self, event):
#     from numpy import mean as np_mean
#     from numpy import array as np_array
#     from numpy import roll as np_roll
#     from numpy.linalg import norm as np_distance
#     from ..utils import closest_pts_to_lines
#
#     pt = self.get_mouse_pos(event)
#     pt_np = np.array([pt], dtype='float32')
#     u = self.umeshes[0]
#     uv = u.uv
#
#     min_pt = ()
#     min_dist = float('inf')
#
#     for f in u.bm.faces:
#         l_a = np_array([crn[uv].uv.to_tuple() for crn in f.loops], dtype='float32')
#         l_b = np_roll(l_a, shift=1, axis=0)
#
#         closest_points = closest_pts_to_lines(pt_np, l_a, l_b)
#         distance = np_distance(pt_np - closest_points, axis=1)
#
#         min_index = np.argmin(distance)
#         if distance[min_index] < min_dist:
#             min_dist = distance[min_index]
#             min_pt = closest_points[min_index]
#
#         face_center = np_mean(l_a, axis=0)
#         distance_face_center = np_distance(pt_np - face_center, axis=1)
#
#         if distance_face_center < min_dist:
#             min_dist = distance_face_center
#             min_pt = face_center
#
#     self.points = (min_pt,)
