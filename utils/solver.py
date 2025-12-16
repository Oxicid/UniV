import numpy as np


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

import unittest

# class TestSolver(__import__('unittest').TestCase):
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
        import unittest
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        result.wasSuccessful()
