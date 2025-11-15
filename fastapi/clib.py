# SPDX-FileCopyrightText: 2025 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later


import platform
import ctypes
from .. import utils
from ctypes import (
    c_int,
    c_double,
    c_void_p,
    c_bool
)

class FastAPI:
    lib: ctypes.CDLL | None = None

    @classmethod
    def load(cls):
        cls.lib = utils.load_lib('univ_fastapi')
        cls.init_linear_solver()

    @classmethod
    def close(cls):
        if cls.lib is None:
            return

        if platform.system() == 'Windows':
            ctypes.windll.kernel32.FreeLibrary(ctypes.c_void_p(cls.lib._handle))
        cls.lib = None

    @classmethod
    def init_linear_solver(cls):
        lib = cls.lib

        # LinearSolver* solver_create(int num_rows, int num_variables, bool least_squares)
        lib.solver_create.argtypes = (c_int, c_int, c_bool)
        lib.solver_create.restype = c_void_p

        # void solver_delete(LinearSolver* solver)
        lib.solver_delete.argtypes = (c_void_p,)
        lib.solver_delete.restype = None

        # void solver_matrix_add(LinearSolver* solver, int row, int col, double value)
        lib.solver_matrix_add.argtypes = (c_void_p, c_int, c_int, c_double)
        lib.solver_matrix_add.restype = None

        # void solver_matrix_add_angles(LinearSolver* solver, int row, double a1, double a2, double a3, int v1_id, int v2_id, int v3_id)
        lib.solver_matrix_add_angles.argtypes = (c_void_p, c_int, c_double, c_double, c_double, c_int, c_int, c_int)
        lib.solver_matrix_add_angles.restype = None

        # void solver_right_hand_side_add(LinearSolver* solver, int index, double value)
        lib.solver_right_hand_side_add.argtypes = (c_void_p, c_int, c_double)
        lib.solver_right_hand_side_add.restype = None

        # void solver_lock_variable(LinearSolver* solver, int index, double value)
        lib.solver_lock_variable.argtypes = (c_void_p, c_int, c_double)
        lib.solver_lock_variable.restype = None

        # double solver_variable_get(LinearSolver* solver, int index)
        lib.solver_variable_get.argtypes = (c_void_p, c_int)
        lib.solver_variable_get.restype = c_double

        # void solver_variable_set(LinearSolver* solver, int index, double value)
        lib.solver_variable_set.argtypes = (c_void_p, c_int, c_double)
        lib.solver_variable_set.restype = None

        # bool solver_solve(LinearSolver* solver)
        lib.solver_solve.argtypes = (c_void_p,)
        lib.solver_solve.restype = c_bool


class LinearSolver:
    @classmethod
    def new(cls, num_rows: int, num_variables: int, least_squares=False):
        assert platform.system() == 'Windows'
        return cls(num_rows, num_variables, least_squares)

    def __init__(self, num_rows: int, num_variables: int, least_squares: bool = False):
        self._handle = FastAPI.lib.solver_create(num_rows, num_variables, least_squares)
        if not self._handle:
            raise RuntimeError("Failed to create native LinearSolver")
        self.num_rows = num_rows
        self.num_variables = num_variables

    def matrix_add(self, row: int, col: int, value: float) -> None:
        FastAPI.lib.solver_matrix_add(self._handle, row, col, value)

    def matrix_add_angles(self, row: int, a1: float, a2: float, a3: float, v1_id: int, v2_id: int, v3_id: int) -> None:
        FastAPI.lib.solver_matrix_add_angles(self._handle, row, a1, a2, a3, v1_id, v2_id, v3_id)

    def right_hand_side_add(self, index: int, value: float) -> None:
        FastAPI.lib.solver_right_hand_side_add(self._handle, index, value)

    def lock_variable(self, index: int, value: float) -> None:
        FastAPI.lib.solver_lock_variable(self._handle, index, value)

    def variable_get(self, index: int) -> float:
        return FastAPI.lib.solver_variable_get(self._handle, index)

    def variable_set(self, index: int, value: float) -> None:
        FastAPI.lib.solver_variable_set(self._handle, index, value)

    def solve(self) -> bool:
        return FastAPI.lib.solver_solve(self._handle)

    def close(self) -> None:
        if self._handle:
            try:
                FastAPI.lib.solver_delete(self._handle)
            finally:
                self._handle = None

    # context manager support
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ensure resource freed
    def __del__(self):
        # destructor may be called during interpreter shutdown; guard against exceptions
        try:
            self.close()
        except Exception:  # noqa
            pass