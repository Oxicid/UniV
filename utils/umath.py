# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import math
import typing
import unittest
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

def attenuate_padding(padding: float, size: float, allowed_padding_ratio: float = 0.25, beta: float = 2.0) -> float:
    """
    Compresses total padding (on both sides) depending on size.

    padding: total padding (left + right)
    size: target size along the axis (width or height)
    allowed_padding_ratio: the proportion of the size that we leave untouched
    beta: attenuation force — greater -> stronger reduce excess padding
    """

    r = padding / size

    if r <= allowed_padding_ratio:
        return padding

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

class LinearSolver:
    class Coeff:
        __slots__ = ('index', 'value')
        def __init__(self, index=0, value=0.0):
            self.index = index
            self.value = value

    class Variable:
        __slots__ = ('index', 'locked', 'value', 'coeffs')
        def __init__(self):
            self.index = -1     # compact index (or ~0 if locked)
            self.locked = False
            self.value = 0.0
            self.coeffs = []    # list of Coeff (row index, value)

    @classmethod
    def new(cls, num_rows: int, num_variables: int, least_squares=False):
        try:
            from .. import fastapi
            from .. import preferences
            if preferences.prefs().use_fastapi and fastapi.FastAPI.lib:
                import platform
                if platform.system() == 'Windows':
                    return fastapi.LinearSolver.new(num_rows, num_variables, least_squares)
        except:  # noqa
            pass

        return cls(num_rows, num_variables, least_squares)

    def __init__(self, num_rows: int, num_variables: int, least_squares=False):
        # num_rows == original number of rows provided by caller
        self.num_rows = num_rows
        self.num_variables = num_variables
        self.least_squares = least_squares

        self.state = 'VARIABLES_CONSTRUCT'

        # storage used during 'matrix construct' stage
        self.M_triplets: list[(int, int, float)] = []   # list of (row, col, value)
        self.b = None         # will be list of rhs vectors after ensure_matrix_construct
        self.x = None         # solution vectors after ensure_matrix_construct

        # compact sizes filled by ensure_matrix_construct
        self.m = 0
        self.n = 0

        self.variable = [self.Variable() for _ in range(num_variables)]

        self._ensure_matrix_construct()

    def _ensure_matrix_construct(self):
        assert self.state == 'VARIABLES_CONSTRUCT'

        # assign compact indices for non-locked variables
        # TODO: Remove???
        n = 0
        for i in range(self.num_variables):
            var = self.variable[i]
            if not var.locked:
                var.index = n
                n += 1

        # m is either given num_rows or n when num_rows == 0
        m = self.num_rows if self.num_rows != 0 else n

        self.m = m
        self.n = n

        assert (self.least_squares or (self.m == self.n)), "For non-least-squares m must equal n"

        # reserve structures
        self.M_triplets = []
        # b and x are lists of numpy vectors (one per rhs)
        self.b = np.zeros(self.m, dtype=np.float64)
        self.x = np.zeros(self.n, dtype=np.float64)

        # TODO: Remove???
        # move variable initial values into x vectors for unlocked variables
        for i in range(self.num_variables):
            v = self.variable[i]
            if not v.locked:
                idx = v.index
                # if user had pre-set variable.value, store into x
                self.x[idx] = v.value

        self.state = 'MATRIX_CONSTRUCT'

    def matrix_add(self, row: int, col: int, value: float):
        # assert self.state == 'MATRIX_CONSTRUCT'

        # if not least_squares and variable[row] is locked -> ignore
        if (not self.least_squares) and self.variable[row].locked:
            return

        if self.variable[col].locked:
            # store coefficient to variable[col].coeffs
            # in non-lsq case the row must be mapped to compact index
            r = row
            if not self.least_squares:
                r = self.variable[row].index
            coeff = LinearSolver.Coeff(r, value)
            self.variable[col].coeffs.append(coeff)
        else:
            # add triplet to matrix (map row and col to compact indices when needed)
            r = row
            if not self.least_squares:
                r = self.variable[row].index
            c = self.variable[col].index

            if r is None or c is None:
                raise IndexError(f'Invalid indices: {row=}, {col=}')
            self.M_triplets.append((r, c, value))


    def matrix_add_angles(self, row: int, a1: float, a2: float, a3: float, v1_id: int, v2_id: int, v3_id: int):
        from math import sin, cos
        v1_id *= 2
        v2_id *= 2
        v3_id *= 2

        sina1: float = sin(a1)
        sina2: float = sin(a2)
        sina3: float = sin(a3)
        sin_max: float = max(sina1, sina2, sina3)
        # Shift vertices to find most stable order.
        if sina3 != sin_max:
            # shift right
            v1_id, v2_id, v3_id = v3_id, v1_id, v2_id
            a1, a2, a3 = a3, a1, a2
            sina1, sina2, sina3 = sina3, sina1, sina2

            if sina2 == sin_max:
                # shift right
                v1_id, v2_id, v3_id = v3_id, v1_id, v2_id
                a1, a2, a3 = a3, a1, a2
                sina1, sina2, sina3 = sina3, sina1, sina2
        # Angle based lscm formulation.
        ratio: float = sina2 / sina3 if sina3 else 1.0  # safe divide
        cosine: float = cos(a1) * ratio
        sine: float = sina1 * ratio

        self.matrix_add(row, v1_id, cosine - 1.0)
        self.matrix_add(row, v1_id + 1, -sine)
        self.matrix_add(row, v2_id, -cosine)
        self.matrix_add(row, v2_id + 1, sine)
        self.matrix_add(row, v3_id, 1.0)

        row += 1
        self.matrix_add(row, v1_id, sine)
        self.matrix_add(row, v1_id + 1, cosine - 1.0)
        self.matrix_add(row, v2_id, -sine)
        self.matrix_add(row, v2_id + 1, -cosine)
        self.matrix_add(row, v3_id + 1, 1.0)

    def right_hand_side_add(self, index: int, value: float):
        if self.least_squares:
            self.b[index] += value
        else:
            # only add if variable not locked
            if not self.variable[index].locked:
                assert self.variable[index].index == index
                self.b[index] += value

    def lock_variable(self, index: int, value: float):
        v = self.variable[index]
        v.locked = True
        v.value = value

    def solve(self):
        # nothing to solve
        assert self.state == 'MATRIX_CONSTRUCT'

        if self.m == 0 or self.n == 0:
            raise ValueError("Empty matrix")

        if not self.M_triplets:
            raise ValueError("No triplets to build matrix")

        # Build dense matrix from triplets
        M = np.zeros((self.m, self.n), dtype=np.float64)
        for (r, c, val) in self.M_triplets:
            M[r, c] += val

        if self.least_squares:
            MtM = M.T @ M
            A = MtM
        else:
            A = M

        # Creating the RHS vector and applying locked variables
        b_vec = self.b.copy()  # TODO: Remove ???
        for i in range(self.num_variables):
            var = self.variable[i]
            if var.locked and var.coeffs:
                for coeff in var.coeffs:
                    # coeff.index already mapped according to earlier logic in matrix_add
                    if 0 <= coeff.index < len(b_vec):  # TODO: Remove len(b_vec) ???
                        b_vec[coeff.index] -= coeff.value * var.value

        # Solve
        if self.least_squares:
            rhs_vec = M.T @ b_vec
            try:
                # Solve (MtM) x = M^T b
                x_compact = np.linalg.solve(A, rhs_vec)
            except np.linalg.LinAlgError:
                # fallback to least-squares solution if singular
                x_compact, *_ = np.linalg.lstsq(A, rhs_vec , rcond=None)
        else:
            try:
                x_compact = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                # fallback to least-squares solution if singular
                x_compact, *_ = np.linalg.lstsq(A, b_vec, rcond=None)

        # store into solver.x
        self.x[:] = x_compact

        # map back solution from compact x into variables
        for i in range(self.num_variables):
            v = self.variable[i]
            if not v.locked:
                idx = v.index
                if 0 <= idx < self.n:
                    v.value = self.x[idx]
            else:
                # locked variable retains its set value
                pass

        self.b.fill(0.0)  # TODO: Remove ???

        self.state = 'MATRIX_SOLVED'
        return True

    def variable_get(self, index: int):
        return self.variable[index].value

    def variable_set(self, index: int, value: float):
        self.variable[index].value = float(value)

    def __str__(self):
        return str([str(v) for v in self.variable])



class TestSolver(unittest.TestCase):
    def test_solver_simple(self):
        # from mathutils import Solver as LinearSolver
        def fill_solver():
            # system:  2*x0 + 3*x1 = 5
            #            x0 -  x1 = 1
            solver.matrix_add(0, 0, 2)
            solver.matrix_add(0, 1, 3)
            solver.matrix_add(1, 0, 1)
            solver.matrix_add(1, 1, -1)

            solver.right_hand_side_add(0, 5)
            solver.right_hand_side_add(1, 1)

            solver.solve()

        solver = LinearSolver(2, 2, least_squares=False)
        fill_solver()

        self.assertAlmostEqual(solver.variable_get(0), 1.6, delta=1e-9)
        self.assertAlmostEqual(solver.variable_get(1), 0.6, delta=1e-9)


        solver = LinearSolver(2, 2, least_squares=True)
        solver.lock_variable(1, 22)  # lock x1 = 22
        fill_solver()

        self.assertAlmostEqual(solver.variable_get(0), -19.8, delta=1e-9)
        self.assertAlmostEqual(solver.variable_get(1), 22.0, delta=1e-9)


    def test_solver_least_squares_with_lock(self):
        # from mathutils import Solver as LinearSolver
        n_faces = 1
        n_verts = 3
        solver = LinearSolver(2 * n_faces, 2 * n_verts, least_squares=True)

        solver.lock_variable(2, -0.726492702960968)
        solver.lock_variable(3, 0.043564677238464355)

        solver.lock_variable(4, 0.771076500415802)
        solver.lock_variable(5, 0.043564677238464355)

        solver.matrix_add(0, 0, -0.5412585554468488)
        solver.matrix_add(0, 1, -0.7496441899404173)
        solver.matrix_add(0, 2, -0.4587414445531512)
        solver.matrix_add(0, 3, 0.7496441899404173)
        solver.matrix_add(0, 4, 1.0)

        solver.matrix_add(1, 0, 0.7496441899404173)
        solver.matrix_add(1, 1, -0.5412585554468488)
        solver.matrix_add(1, 2, -0.7496441899404173)
        solver.matrix_add(1, 3, -0.4587414445531512)
        solver.matrix_add(1, 5, 1.0)

        solver.solve()

        self.assertAlmostEqual(solver.variable_get(0), 0.22162558147294173, delta=1e-9)
        self.assertAlmostEqual(solver.variable_get(1), 1.3567104116562398, delta=1e-9)

        self.assertAlmostEqual(solver.variable_get(2), -0.726492702960968, delta=1e-9)
        self.assertAlmostEqual(solver.variable_get(3), 0.043564677238464355, delta=1e-9)

        self.assertAlmostEqual(solver.variable_get(4), 0.771076500415802, delta=1e-9)
        self.assertAlmostEqual(solver.variable_get(5), 0.043564677238464355, delta=1e-9)

    @classmethod
    def start(cls):
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        result.wasSuccessful()
