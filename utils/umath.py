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
    def intersect_point_line_segment(pt, line_a, line_b):
        close_pt, percent = intersect_point_line(pt, line_a, line_b)
        if percent < 0.0:
            close_pt = line_a
        elif percent > 1.0:
            close_pt = line_b
        return close_pt, (close_pt - pt).length

class LinearSolver:
    def __init__(self, m, n, least_squares=False):
        self.m = m
        self.n = n
        self.least_squares = least_squares
        self.M = np.zeros((m, n), dtype=np.float64)
        self.b = []      # list of right parts (vectors)
        self.x = []      # solvers
        self.locked: list[bool] = []
        self.values = []  # fixed values

    def add_rhs(self, b):
        self.b.append(np.array(b, dtype=np.float64))
        self.x.append(np.zeros(self.n, dtype=np.float64))

    def solve(self):
        if self.m == 0 or self.n == 0:
            return True

        # creating a copy of the matrix (so as not to spoil the original)
        M = self.M.copy()

        # apply locks
        for j, locked in enumerate(self.locked):
            if locked:
                # zero the column and set the diagonal=1
                M[:, j] = 0.0
                M[j, j] = 1.0

        # Least squares
        if self.least_squares:
            Mt = M.T
            MtM = Mt @ M

        success = True
        for k, b in enumerate(self.b):
            b = b.copy()
            # for locked variables, replace b[j] = value
            for j, locked in enumerate(self.locked):
                if locked:
                    b[j] = self.values[j]

            try:
                if self.least_squares:
                    Mtb = Mt @ b  # noqa pylint: disable=used-before-assignment
                    x = np.linalg.solve(MtM, Mtb)  # noqa pylint: disable=used-before-assignment
                else:
                    x = np.linalg.solve(M, b)
                self.x[k] = x
            except np.linalg.LinAlgError:
                success = False
        return success