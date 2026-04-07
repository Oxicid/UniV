/* SPDX-FileCopyrightText: 2026 Oxicid
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */


/** \file
 * \ingroup bmesh
 *
 * BM construction functions.
 */


#include "BKE_customdata.hh"
#include "DNA_mesh_types.h"

#include "bmesh.hh"


namespace blender {
	
bool BM_verts_from_edges(BMVert **vert_arr, BMEdge **edge_arr, int len) {return false;}
bool BM_edges_from_verts(BMEdge **edge_arr, BMVert **vert_arr, const int len)  {return false;}

void BM_edges_from_verts_ensure(BMesh *bm, BMEdge **edge_arr, BMVert **vert_arr, const int len) {}
void BM_elem_attrs_copy(BMesh *bm, const BMCustomDataCopyMap &map, const BMFace *src, BMFace *dst) {}

void BM_elem_attrs_copy(BMesh *bm, const BMCustomDataCopyMap &map, const BMLoop *src, BMLoop *dst) {}

void BM_elem_attrs_copy(BMesh *bm, const BMVert *src, BMVert *dst) {}
void BM_elem_attrs_copy(BMesh *bm, const BMEdge *src, BMEdge *dst) {}
void BM_elem_attrs_copy(BMesh *bm, const BMFace *src, BMFace *dst) {}
void BM_elem_attrs_copy(BMesh *bm, const BMLoop *src, BMLoop *dst) {}
void BM_elem_select_copy(BMesh *bm_dst, void *ele_dst_v, const void *ele_src_v) {}

BMFace *BM_face_create_ngon(BMesh *bm,
                            BMVert *v1,
                            BMVert *v2,
                            BMEdge **edges,
                            const int len,
                            const BMFace *f_example,
                            const eBMCreateFlag create_flag)
							{return nullptr;}

}  // namespace blender