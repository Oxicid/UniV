/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

/**##################
   Min solver requirements **/
#include "eigen_capi.h"
#include "BLI_math_vector.h"
/* ################## */


#include <stdio.h>
#include <iostream>
#include "BLI_compiler_attrs.h"

#include "BLI_utildefines.h"


#include "BLI_alloca.h"
#include "BLI_linklist.h"
#include "BLI_math_base.hh"
#include "BLI_math_inline.h"
#include "BLI_math_geom.h"
#include "BLI_math_matrix.h"
#include "BLI_memarena.h"
#include "BLI_polyfill_2d.h"
#include "BLI_polyfill_2d_beautify.h"

#include "bmesh.hh"

using namespace blender;

/* This counter should only be changed when new features are added or critical changes are made. */
#define FASTAPI_VERSION 3

# define VERTICAL_CONSTR 2
# define HORIZONTAL_CONSTR 3


#ifdef _WIN32
#  define DLL_EXPORT __declspec(dllexport)
#else
#  define DLL_EXPORT
#endif


static inline bool BMesh_is_full_face_selected(BMesh *bm){
    if (bm->totfacesel) {
        return bm->totfacesel == bm->totface;
	}
    return false;
}

static inline bool BMesh_is_full_face_deselected(BMesh *bm){
        return bm->totfacesel == 0;
}


static inline void write_uv_line(float **ptr, BMLoop *l, const int offsets)
{
  float *dst = *ptr;

  float *uv = BM_ELEM_CD_GET_FLOAT_P(l, offsets);
  dst[0] = uv[0];
  dst[1] = uv[1];

  uv = BM_ELEM_CD_GET_FLOAT_P(l->next, offsets);
  dst[2] = uv[0];
  dst[3] = uv[1];

  *ptr += 4;
}


extern "C" {
DLL_EXPORT int version()
{
  return FASTAPI_VERSION;
}


DLL_EXPORT void UniV_extract_data_constraints2d(
    BMesh *bm,
    const int uv_offset,
    const int constr_offset,
    const bool sync,
    float *r_varray,
    float *r_harray,
    int *r_tot_v,
    int *r_tot_h)
{
    int total_vlines = 0;
    int total_hlines = 0;

    float *vptr = r_varray;
    float *hptr = r_harray;

    const CustomData *data = &bm->edata;

    if (uv_offset == -1) {
      std::cout << "UniV: FastAPI: UniV_extract_data_constraints2d: Can't get uv layer index\n";
      return;
    }

    BMEdge *e;
    BMLoop *l;
    BMIter iter, liter;

    bool check_hidden = false;
    bool check_select = false;

    if (!BMesh_is_full_face_selected(bm)) {

        if (sync) {
            check_hidden = true;
        }
        else {
            if (BMesh_is_full_face_deselected(bm))
                return;

            check_select = true;
        }
    }

    BM_ITER_MESH (e, &iter, bm, BM_EDGES_OF_MESH) {

        void *value = POINTER_OFFSET(e->head.data, constr_offset);

        int edge_idx = *(int *)value;

        if (!edge_idx)
            continue;

        BM_ITER_ELEM (l, &liter, e, BM_LOOPS_OF_EDGE) {

            if ((check_hidden && BM_elem_flag_test(l->f, BM_ELEM_HIDDEN)) ||
				(check_select && !BM_elem_flag_test(l->f, BM_ELEM_SELECT))) {
                edge_idx >>= 2;
				continue;
				}

            int bits = edge_idx & 3;

            if (bits == VERTICAL_CONSTR) {
              write_uv_line(&vptr, l, uv_offset);
                total_vlines++;
            }
            else if (bits == HORIZONTAL_CONSTR) {
              write_uv_line(&hptr, l, uv_offset);
                total_hlines++;
            }

        	edge_idx >>= 2;
        }
    }

    *r_tot_v = total_vlines * 2;
    *r_tot_h = total_hlines * 2;
}

DLL_EXPORT int UniV_extract_data_seams2d(
    BMesh *bm,
    const int uv_offset,
    const bool sync,
    float *r_array)
{
    int total_lines = 0;

    float *arrptr = r_array;

    if (uv_offset == -1) {
      std::cout << "UniV: FastAPI: UniV_extract_data_seams2d: Can't get uv layer index\n";
      return 0;
    }

    BMEdge *e;
    BMLoop *l;
    BMIter iter, liter;

    bool check_hidden = false;
    bool check_select = false;

    if (!BMesh_is_full_face_selected(bm)) {

        if (sync) {
            check_hidden = true;
        }
        else {
            if (BMesh_is_full_face_deselected(bm))
                return 0;

            check_select = true;
        }
    }

    BM_ITER_MESH (e, &iter, bm, BM_EDGES_OF_MESH) {

        if (BM_elem_flag_test(e, BM_ELEM_SEAM)) {

			BM_ITER_ELEM (l, &liter, e, BM_LOOPS_OF_EDGE) {

				if ((check_hidden && BM_elem_flag_test(l->f, BM_ELEM_HIDDEN)) ||
					(check_select && !BM_elem_flag_test(l->f, BM_ELEM_SELECT))) {
					continue;
					}

				write_uv_line(&arrptr, l, uv_offset);
				total_lines++;
			}
		}
	}
	return total_lines * 2;
}

//DLL_EXPORT void UniV_calc_tessellation_for_face_impl(std::array<BMLoop *, 3> *looptris,
//                                                      BMFace *efa,
//                                                      MemArena **pf_arena_p,
//                                                      const bool face_normal)
//
//{
//	UniV_calc_tessellation_for_face_impl(looptris, efa, pf_arena_p, false);
//
//}
//
//DLL_EXPORT void UniV_polyfill_calc(const float (*coords)[2],
//                       unsigned int coords_num,
//                       int coords_sign,
//                       unsigned int (*r_tris)[3]) {
//    BLI_polyfill_calc(coords, coords_num, 1, r_tris);
//}


DLL_EXPORT LinearSolver* solver_create(int num_rows, int num_variables, bool least_squares)
{
    if (!least_squares)
        return EIG_linear_solver_new(num_rows, num_variables, 1);
    else
        return EIG_linear_least_squares_solver_new(num_rows, num_variables, 1);
}

DLL_EXPORT void solver_delete(LinearSolver* solver)
{
    EIG_linear_solver_delete(solver);
}

DLL_EXPORT void solver_matrix_add(LinearSolver* solver, int row, int col, double value)
{
    EIG_linear_solver_matrix_add(solver, row, col, value);
}

DLL_EXPORT void solver_matrix_add_angles(LinearSolver* solver, int row, double a1, double a2, double a3, int v1_id, int v2_id, int v3_id)
{
    double sina1 = sin(a1);
    double sina2 = sin(a2);
    double sina3 = sin(a3);

    const double sinmax = max_ddd(sina1, sina2, sina3);

    /* Shift vertices to find most stable order. */
    if (sina3 != sinmax) {
      SHIFT3(int, v1_id, v2_id, v3_id);
      SHIFT3(double, a1, a2, a3);
      SHIFT3(double, sina1, sina2, sina3);

      if (sina2 == sinmax) {
        SHIFT3(int, v1_id, v2_id, v3_id);
        SHIFT3(double, a1, a2, a3);
        SHIFT3(double, sina1, sina2, sina3);
      }
    }

    /* Angle based lscm formulation. */
    const double ratio = (sina3 == 0.0f) ? 1.0f : sina2 / sina3;
    const double cosine = cos(a1) * ratio;
    const double sine = sina1 * ratio;

    EIG_linear_solver_matrix_add(solver, row, 2 * v1_id, cosine - 1.0f);
    EIG_linear_solver_matrix_add(solver, row, 2 * v1_id + 1, -sine);
    EIG_linear_solver_matrix_add(solver, row, 2 * v2_id, -cosine);
    EIG_linear_solver_matrix_add(solver, row, 2 * v2_id + 1, sine);
    EIG_linear_solver_matrix_add(solver, row, 2 * v3_id, 1.0);
    row++;

    EIG_linear_solver_matrix_add(solver, row, 2 * v1_id, sine);
    EIG_linear_solver_matrix_add(solver, row, 2 * v1_id + 1, cosine - 1.0f);
    EIG_linear_solver_matrix_add(solver, row, 2 * v2_id, -sine);
    EIG_linear_solver_matrix_add(solver, row, 2 * v2_id + 1, -cosine);
    EIG_linear_solver_matrix_add(solver, row, 2 * v3_id + 1, 1.0);
}

DLL_EXPORT void solver_right_hand_side_add(LinearSolver* solver, int index, double value)
{
    EIG_linear_solver_right_hand_side_add(solver, 0, index, value);
}

DLL_EXPORT void solver_lock_variable(LinearSolver* solver, int index, double value)
{
    EIG_linear_solver_variable_lock(solver, index);
    EIG_linear_solver_variable_set(solver, 0, index, value);
}

DLL_EXPORT double solver_variable_get(LinearSolver* solver, int index)
{
    return EIG_linear_solver_variable_get(solver, 0, index);
}

DLL_EXPORT void solver_variable_set(LinearSolver* solver, int index, double value)
{
    EIG_linear_solver_variable_set(solver, 0, index, value);
}

DLL_EXPORT bool solver_solve(LinearSolver* solver)
{
    return EIG_linear_solver_solve(solver);
}

} // extern "C"