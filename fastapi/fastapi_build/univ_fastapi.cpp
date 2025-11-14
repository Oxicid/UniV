/* SPDX-FileCopyrightText: 2025 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */


#pragma once
// #include "BLI_memory_utils.hh"
#include "eigen_capi.h"
#include "BLI_math_vector.h"



#ifdef _WIN32
#  define DLL_EXPORT __declspec(dllexport)
#else
#  define DLL_EXPORT
#endif


extern "C" {

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