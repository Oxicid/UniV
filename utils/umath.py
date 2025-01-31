# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import numpy as np
from bl_math import lerp
from mathutils import Vector

def all_equal(sequence):
    sequence_iter = iter(sequence)
    first = next(sequence_iter)
    for i in sequence_iter:
        if i != first:
            return False
    return True

def vec_isclose(a, b, abs_tol: float = 0.00001):
    return all(math.isclose(a1, b1, abs_tol=abs_tol) for a1, b1 in zip(a, b))

def vec_isclose_to_uniform(delta: Vector, abs_tol: float = 0.00001):
    return all(math.isclose(component, 1.0, abs_tol=abs_tol) for component in delta)

def vec_isclose_to_zero(delta: Vector, abs_tol: float = 0.00001):
    return all(math.isclose(component, 0.0, abs_tol=abs_tol) for component in delta)

# Source: https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56
def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linear Interpolation, get the fraction between a and b on which v resides.
    Examples
    --------
        0.5 == inv_lerp(0, 100, 50)
        0.8 == inv_lerp(1, 5, 4.2)
    """
    return (v - a) / (b - a)

def remap(i_min: float, i_max: float, o_min: float, o_max: float, v: float) -> float:
    """Remap values from one linear scale to another, a combination of lerp and inv_lerp.
    i_min and i_max are the scale on which the original value resides,
    o_min and o_max are the scale to which it should be mapped.
    Examples
    --------
        45 == remap(0, 100, 40, 50, 50)
        6.2 == remap(1, 5, 3, 7, 4.2)
    """
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

def round_vector_to_cardinal(v):
    x, y = v
    if abs(x) >= abs(y):
        return Vector([np.sign(x), 0])
    else:
        return Vector([0, np.sign(y)])

def round_threshold(a, min_clip):
    return round(float(a) / min_clip) * min_clip

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
