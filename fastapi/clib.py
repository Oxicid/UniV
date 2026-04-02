# SPDX-FileCopyrightText: 2025 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import platform
import ctypes
from .. import utypes
from .. import utils
from ctypes import (
    c_int,
    c_float,
    c_double,
    c_void_p,
    c_bool,
    POINTER,
    byref
)

class FastAPI:
    lib: ctypes.CDLL | None = None
    _expected_fastapi_min_version = 3
    @classmethod
    def load(cls):
        cls.lib = utils.load_lib('univ_fastapi')
        if hasattr(cls.lib, 'version'):  # TODO: Delete after 3 month
            if cls.lib.version() < cls._expected_fastapi_min_version:
                cls.close()
                print(f"UniV: FastAPI: Expected minimal version {cls._expected_fastapi_min_version}, given: {cls.lib.version()!r}.")
                return
        else:
            cls.close()
            return
        cls.init_linear_solver()
        cls.init_extract_data()

    @classmethod
    def close(cls):
        if cls.lib is None:
            return

        handle = cls.lib._handle
        cls.lib.UniV_extract_data_constraints2d = None  # decref

        match platform.system():
            case 'Windows':
                dll_close = ctypes.windll.kernel32.FreeLibrary  # noqa
            case "Darwin":
                stdlib = ctypes.CDLL("libc.dylib")
                dll_close = stdlib.dlclose
            case "Linux":
                try:
                    stdlib = ctypes.CDLL("")
                except OSError:
                    # Alpine Linux.
                    stdlib = ctypes.CDLL("libc.so")
                dll_close = stdlib.dlclose
            case platform_:
                raise NotImplementedError(f"Unknown platform: {platform_!r}.")

        dll_close.argtypes = [ctypes.c_void_p]
        res_lib_close = dll_close(ctypes.c_void_p(handle))
        if not res_lib_close:
            print("UniV: FastAPI: Cant unload shared library.")
        import gc
        gc.collect()  # clear for free lib

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

    @classmethod
    def init_extract_data(cls):
        from .. import btypes
        lib = cls.lib

        # void UniV_extract_data_constraints2d(
        #     BMesh *bm,
        #     const int uv_offset,
        #     const int constr_offset,
        #     const bool sync,
        #     float *r_varray,
        #     float *r_harray,
        #     int *r_tot_v,
        #     int *r_tot_h)
        lib.UniV_extract_data_constraints2d.argtypes = (POINTER(btypes.CBMesh),
                                                        c_int,
                                                        c_int,
                                                        c_bool,
                                                        POINTER(c_float),
                                                        POINTER(c_float),
                                                        POINTER(c_int),
                                                        POINTER(c_int)
                                                        )
        lib.UniV_extract_data_constraints2d.restype = None

        # int UniV_extract_data_seams2d(
        #     BMesh *bm,
        #     const int uv_layer,
        #     const bool sync,
        #     float *r_array)

        lib.UniV_extract_data_seams2d.argtypes = (POINTER(btypes.CBMesh),
                                                        c_int,
                                                        c_bool,
                                                        POINTER(c_float)
                                                        )
        lib.UniV_extract_data_seams2d.restype = c_int


class ExtractData:
    @staticmethod
    def extract_constraints_data(umesh: 'utypes.UMesh', attr):
        from .. import btypes

        c_bm: btypes.CBMesh = btypes.PyBMesh.get_fields_from_pyobj(umesh.bm).bm

        uv_offset = ExtractData.get_uv_offset(umesh)
        constr_offset = ExtractData.get_constr_offset(umesh, attr)

        max_data_shape = (c_bm.contents.totloop*2, 2)
        varray = np.empty(max_data_shape, np.float32)
        harray = np.empty(max_data_shape, np.float32)

        tot_v_coords = c_int()
        tot_h_coords = c_int()

        FastAPI.lib.UniV_extract_data_constraints2d(
            c_bm,
            uv_offset,
            constr_offset,
            umesh.sync,
            varray.ctypes.data_as(POINTER(c_float)),
            harray.ctypes.data_as(POINTER(c_float)),
            byref(tot_v_coords),
            byref(tot_h_coords)
        )
        return varray[:tot_v_coords.value], harray[:tot_h_coords.value]

    @staticmethod
    def extract_seams_data(umesh: 'utypes.UMesh'):
        from .. import btypes

        c_bm: btypes.PyBMesh = btypes.PyBMesh.get_fields_from_pyobj(umesh.bm).bm
        total_corners = c_bm.contents.totloop
        uv_offset = ExtractData.get_uv_offset(umesh)

        max_data_shape = (total_corners*2, 2)
        r_array = np.empty(max_data_shape, np.float32)

        tot_coords = FastAPI.lib.UniV_extract_data_seams2d(
            c_bm,
            uv_offset,
            umesh.sync,
            r_array.ctypes.data_as(POINTER(c_float))
        )

        return r_array[:tot_coords]

    @staticmethod
    def get_uv_offset(umesh):
        from .. import btypes
        CD_PROP_FLOAT2 = 49
        n = btypes.BPy_BMLayerItem.get_fields_from_pyobj(umesh.uv).index
        assert(n >= 0)

        custom_data: btypes.CustomData = btypes.PyBMesh.get_fields_from_pyobj(umesh.bm).bm.contents.ldata

        i = custom_data.typemap[CD_PROP_FLOAT2]
        if i != -1:
          # If the value of n goes past the block of layers of the correct type, return -1. */
            if (i + n) < custom_data.totlayer:
                layer = custom_data.layers[i + n]
                if layer.type == CD_PROP_FLOAT2:
                    return layer.offset
        raise
        return -1

    @staticmethod
    def get_constr_offset(umesh, attr):
        from .. import btypes
        py_constr_layer: btypes.BPy_BMLayerItem = btypes.BPy_BMLayerItem.get_fields_from_pyobj(attr)

        n = py_constr_layer.index
        typ = py_constr_layer.type

        custom_data: btypes.CustomData = btypes.PyBMesh.get_fields_from_pyobj(umesh.bm).bm.contents.edata

        assert(n >= 0)

        layer_index = custom_data.typemap[typ]
        assert layer_index != -1

        layer: btypes.CustomDataLayer = custom_data.layers[layer_index + n]
        assert layer.type == typ
        return layer.offset




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

import unittest

class TestExtractData(unittest.TestCase):
    @staticmethod
    def get_umesh_with_constraints_and_seams():
        import bmesh
        bm = bmesh.new()

        v1 = bm.verts.new((0, 1, 0))
        v2 = bm.verts.new((0, 0, 0))
        v3 = bm.verts.new((1, 0, 0))
        v4 = bm.verts.new((0, -1, 0))
        v5 = bm.verts.new((-1, 0, 0))
        v6 = bm.verts.new((0, 0, 1))
        v7 = bm.verts.new((0, 1, 1))

        bm.verts.index_update()

        bm.edges.new((v6, v7))  # Wire Edge

        f1 = bm.faces.new((v1, v2, v3))
        f2 = bm.faces.new((v3, v2, v4))
        f3 = bm.faces.new((v4, v2, v5))

        f4 = bm.faces.new((v2, v4, v6))

        uv = bm.loops.layers.uv.new()
        bm.faces.ensure_lookup_table()

        for crn, uv_co in zip(f1.loops, ((-1, 0), (0, 0), (0, 1))):
            crn[uv].uv = uv_co

        for crn, uv_co in zip(f2.loops, ((0, 1), (0, 0), (1, 0))):
            crn[uv].uv = uv_co

        for crn, uv_co in zip(f3.loops, ((1, 0), (0, 0), (0, -1))):
            crn[uv].uv = uv_co

        for crn, uv_co in zip(f4.loops, ((1, 0), (2, 0), (2, 1))):
            crn[uv].uv = uv_co

        V = '10'
        H = '11'
        ERR = '01'

        atr = bm.edges.layers.int.new('univ_constraints')
        f1.edges[0][atr] = int(H + V, 2)
        f1.edges[1][atr] = int(H + H, 2)
        f1.edges[2][atr] = int(ERR, 2)
        f1.edges[1].seam = True

        f2.edges[1][atr] = int(ERR + H + H, 2)
        f2.edges[1].seam = True

        f3.edges[1][atr] = int(ERR + V, 2)
        f3.edges[1].seam = True

        f4.edges[1][atr] = int(H + V + V, 2)
        f4.edges[1].seam = True

        wire_edge = next(e for e in bm.edges if e.is_wire)
        wire_edge[atr] = int(H, 2)
        wire_edge.seam = True

        u = utypes.UMesh(bm, None, is_edit_bm=False)
        u.sync = True

        return u, atr

    def test_extract_data_constraints(self):


        umesh, attr = self.get_umesh_with_constraints_and_seams()
        varray, harray = ExtractData.extract_constraints_data(umesh, attr)

        self.assertEqual(len(varray), 6)
        self.assertEqual(len(harray), 8)

        expect_varray = [[-1,0],[0,0], [0,0],[0,-1], [2,0],[2,1]]
        expect_harray = [[0,1],[0,0], [0,0],[0,1], [1,0],[2,0], [0,0],[1,0]]


        self.assertEqual(varray.tolist(), expect_varray)
        self.assertEqual(harray.tolist(), expect_harray)

        umesh.free()

    def test_extract_data_seams(self):
        umesh, _ = self.get_umesh_with_constraints_and_seams()
        data = ExtractData.extract_seams_data(umesh)

        self.assertEqual(len(data), 14)

        expect_data = [[0,1],[0,0], [0,0],[0,1], [1,0],[2,0], [0,0],[1,0], [1,0],[0,0], [0,0],[0,-1], [2,0],[2,1]]
        self.assertEqual(data.tolist(), expect_data)

        umesh.free()

    @classmethod
    def start(cls):
        suite = unittest.TestLoader().loadTestsFromTestCase(cls)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        result.wasSuccessful()
        return result