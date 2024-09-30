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
    return math.isclose(delta.x, 1.0, abs_tol=abs_tol) and math.isclose(delta.y, 1.0, abs_tol=abs_tol)

def vec_isclose_to_zero(delta: Vector, abs_tol: float = 0.00001):
    return math.isclose(delta.x, 0, abs_tol=abs_tol) and math.isclose(delta.y, 0, abs_tol=abs_tol)

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

def round_threshold(a, min_clip):
    return round(float(a) / min_clip) * min_clip


def closest_pt_to_line(pt: Vector, l_a: Vector, l_b: Vector):
    line_vec = l_b - l_a
    pt_vec = pt - l_a
    line_len_squared = line_vec.dot(line_vec)

    projection = pt_vec.dot(line_vec) / line_len_squared

    if projection < 0:
        return l_a
    elif projection > 1:
        return l_b
    return l_a + projection * line_vec

def closest_pts_to_lines(pt: np.ndarray, l_a: np.ndarray, l_b: np.ndarray) -> np.ndarray:
    line_vecs = l_b - l_a
    pt_vecs = pt - l_a

    line_len_squared = np.sum(line_vecs ** 2, axis=1)
    projections = np.sum(pt_vecs * line_vecs, axis=1) / line_len_squared

    # Restrict projections in the range [0, 1] for those that are inside segments
    projections_clipped = np.clip(projections, 0, 1)

    closest_pts = l_a + projections_clipped[:, np.newaxis] * line_vecs

    return closest_pts
