/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "bmesh_class.hh"


BMEdge *BM_edge_rotate(BMesh *bm, BMEdge *e, bool ccw, short check_flag);
bool BM_edge_rotate_check(BMEdge *e);