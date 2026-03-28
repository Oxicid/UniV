/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/** \file
 * \ingroup bmesh
 *
 * This file contains functions for locally modifying
 * the topology of existing mesh data. (split, join, flip etc).
 */

#include "BLI_math_vector.h"
#include "BLI_vector.hh"

#include "BKE_customdata.hh"

#include "bmesh.hh"
#include "intern/bmesh_private.hh"

BMEdge *BM_edge_rotate(BMesh *bm, BMEdge *e, bool ccw, short check_flag) {return nullptr;}
bool BM_edge_rotate_check(BMEdge *e) {return false;}