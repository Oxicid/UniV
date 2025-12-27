# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import typing
import mathutils
import numpy as np
from math import floor
from bl_math import lerp
from mathutils import Vector
from mathutils.geometry import intersect_point_line


def all_equal(sequence: typing.Iterable, key: typing.Callable | None = None):
    if key is None:
        sequence_iter = iter(sequence)
        try:
            first = next(sequence_iter)
        except StopIteration:
            return True

        for i in sequence_iter:
            if i != first:
                return False
    else:
        sequence_iter = iter(sequence)
        try:
            first = key(next(sequence_iter))
        except StopIteration:
            return True
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


def safe_divide(a: int | float, b: int | float) -> float:
    return a / b if b else 0.0

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

def wrap_line_nearest(start: float, width: float, min_bound: float, max_bound: float, eps=1e-8) -> float:
    """Returns a position within [min_bound, max_bound - width].
    If the original segment fits within the given segment, returns start.
    Otherwise, returns either the wrap version or the clamped version (closest to start).
    """
    if min_bound <= start and start + width <= max_bound:
        return start

    rng = max_bound - min_bound - width
    if rng <= eps:
        return max(min(start, min_bound + rng), min_bound)

    wrapped = (start - min_bound) % rng + min_bound
    clamped = max(min(start, min_bound + rng), min_bound)
    if abs(wrapped - start) <= abs(clamped - start):
        return wrapped
    else:
        return clamped

def attenuate_padding(pad: float, size: float, allowed_padding_ratio: float = 0.25, beta: float = 2.0) -> float:
    """
    Compresses total padding (on both sides) depending on size.

    padding: total padding (left + right)
    size: target size along the axis (width or height)
    allowed_padding_ratio: the proportion of the size that we leave untouched
    beta: attenuation force — greater -> stronger reduce excess padding
    """

    r = pad / size

    if r <= allowed_padding_ratio:
        return pad

    excess = r - allowed_padding_ratio
    attenuated_excess = excess / (1.0 + beta * excess)

    r_eff = allowed_padding_ratio + attenuated_excess
    # do not allow to exceed the size
    if r_eff >= 0.999999:
        r_eff = 0.999999
    return r_eff * size


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
    near_pt, percent = intersect_point_line(pt, l_a, l_b)
    if percent < 0.0:
        return l_a
    elif percent > 1.0:
        return l_b
    return near_pt


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


def find_nearest(a, v: float) -> float | None:
    idx = np.searchsorted(a, v)
    if not idx:
        if not len(a):
            return None
        return a[0]

    if idx == len(a):
        return a[-1]
    return a[idx] if abs(a[idx] - v) < abs(a[idx - 1] - v) else a[idx - 1]


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
    try:
        from numpy.core.umath_tests import inner1d  # noqa
        return inner1d(a, b)  # x2 faster then einsum, but deprecated
    except: # noqa
        return np.einsum('ij,ij->i', a, b)


def np_vec_normalized(a, keepdims=True):
    return np.linalg.norm(a, axis=1, keepdims=keepdims)


def largest_gap_midpoint_for_hue(values: list[float]):
    if not values:
        return 0.0
    if len(values) == 1:
        return 0.65 if values[0] < 0.5 else 0.35

    values.insert(0, 0.0)
    values.append(1.0)
    values.sort()

    mid = 0.5
    max_gap = -1

    for i in range(len(values) - 1):
        a = values[i]
        b = values[i+1]

        if (gap := (b - a)) > max_gap:
            max_gap = gap
            mid = (a + b) / 2
    return mid


def pack_rgba_to_uint32(rgba) -> int:
    r = int(rgba[0] * 255.0) & 0xFF
    g = int(rgba[1] * 255.0) & 0xFF
    b = int(rgba[2] * 255.0) & 0xFF
    a = int(rgba[3] * 255.0) & 0xFF
    return (r << 24) | (g << 16) | (b << 8) | a

if hasattr(mathutils.geometry, 'intersect_point_line_segment'):
    # version >= (5, 0, 0)
    intersect_point_line_segment = mathutils.geometry.intersect_point_line_segment
else:
    def intersect_point_line_segment(pt: Vector, line_a: Vector, line_b: Vector) -> tuple[Vector, float]:
        close_pt, percent = intersect_point_line(pt, line_a, line_b)
        if percent < 0.0:
            close_pt = line_a
        elif percent > 1.0:
            close_pt = line_b
        return close_pt, (close_pt - pt).length

UNIT_CONVERTION = {
    'mm': (0.001, 1000),
    'cm': (0.01, 100),
    'm': (1, 1),
    'km': (1000, 0.001),
    'in': (0.0254, 39.3701),
    'ft': (0.3048, 3.28084),
    'yd': (0.9144, 1.09361),
    'mi': (1609.34, 0.000621371),
}

UNITS = '(mi|mm|cm|m|km|in|ft|yd)'
UNITS_T = typing.Literal['mm', 'cm', 'm', 'km', 'in', 'ft', 'yd', 'mi',]


def unit_conversion(value: float, from_type: UNITS_T, to_type: UNITS_T) -> float:
    return value * UNIT_CONVERTION[from_type.lower()][0] * UNIT_CONVERTION[to_type.lower()][1]
