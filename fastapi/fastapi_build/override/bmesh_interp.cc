/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "bmesh_interp.hh"
#include "bmesh.hh"

void BM_loop_interp_multires_ex(BMesh * /*bm*/,
                                BMLoop *l_dst,
                                const BMFace *f_src,
                                const float f_dst_center[3],
                                const float f_src_center[3],
                                const int cd_loop_mdisp_offset) {}


void BM_face_interp_multires_ex(BMesh *bm,
                                BMFace *f_dst,
                                const BMFace *f_src,
                                const float f_dst_center[3],
                                const float f_src_center[3],
                                int cd_loop_mdisp_offset) {}
								
								