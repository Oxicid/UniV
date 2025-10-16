# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa
import typing

from bmesh.types import BMesh, BMFace, BMEdge, BMVert, BMLoop

from .. import utypes

USE_GENERIC_UV_SYNC = hasattr(BMesh, 'uv_select_sync_valid')

if USE_GENERIC_UV_SYNC:
    def calc_selected_uv_faces(umesh: 'utypes.UMesh') -> list[BMFace] | typing.Sequence[BMFace]:
        if umesh.is_full_face_deselected:
            return []

        if umesh.sync:
            if umesh.is_full_face_selected:
                if umesh.is_full_face_selected_for_avoid_force_explicit_check:
                    return umesh.bm.faces
                if not umesh.sync_valid:
                    return [f for f in umesh.bm.faces if f.select]

                return [f for f in umesh.bm.faces if f.uv_select]
            return [f for f in umesh.bm.faces if f.uv_select and not f.hide]

        if umesh.is_full_face_selected:
            return [f for f in umesh.bm.faces if f.uv_select]
        return [f for f in umesh.bm.faces if f.uv_select and f.select]
else:
    def calc_selected_uv_faces(umesh: 'utypes.UMesh') -> list[BMFace] | typing.Sequence[BMFace]:
        if umesh.is_full_face_deselected:
            return []

        if umesh.sync:
            if umesh.is_full_face_selected:
                return umesh.bm.faces
            return [f for f in umesh.bm.faces if f.select]

        uv = umesh.uv
        if umesh.is_full_face_selected:
            if umesh.elem_mode == 'VERT':
                return [f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops)]
            else:
                return [f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops)]
        if umesh.elem_mode == 'VERT':
            return [f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops) and f.select]
        else:
            return [f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops) and f.select]

if USE_GENERIC_UV_SYNC:
    def calc_selected_uv_faces_iter(umesh: 'utypes.UMesh') -> 'typing.Generator[BMFace] | typing.Sequence':
        if umesh.is_full_face_deselected:
            return []

        if umesh.sync:
            if umesh.is_full_face_selected:
                if umesh.is_full_face_selected_for_avoid_force_explicit_check:
                    return umesh.bm.faces
                if not umesh.sync_valid:
                    return (f for f in umesh.bm.faces if f.select)

                return (f for f in umesh.bm.faces if f.uv_select)
            return (f for f in umesh.bm.faces if f.uv_select and not f.hide)

        if umesh.is_full_face_selected:
            return (f for f in umesh.bm.faces if f.uv_select)
        return (f for f in umesh.bm.faces if f.uv_select and f.select)
else:
    def calc_selected_uv_faces_iter(umesh: 'utypes.UMesh') -> 'typing.Generator[BMFace] | typing.Sequence':
        if umesh.is_full_face_deselected:
            return ()

        if umesh.sync:
            if umesh.is_full_face_selected:
                return umesh.bm.faces
            return (f for f in umesh.bm.faces if f.select)

        uv = umesh.uv
        if umesh.is_full_face_selected:
            if umesh.elem_mode == 'VERT':
                return (f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops))
            else:
                return (f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops))
        if umesh.elem_mode == 'VERT':
            return (f for f in umesh.bm.faces if all(crn[uv].select for crn in f.loops) and f.select)
        else:
            return (f for f in umesh.bm.faces if all(crn[uv].select_edge for crn in f.loops) and f.select)


def calc_selected_verts(umesh: 'utypes.UMesh') -> list[BMVert] | typing.Any:  # noqa
    if umesh.is_full_vert_deselected:
        return []
    if umesh.is_full_vert_selected:
        return umesh.bm.verts
    return [v for v in umesh.bm.verts if v.select]


def calc_selected_edges(umesh: 'utypes.UMesh') -> list[BMEdge] | typing.Any:  # noqa
    if umesh.is_full_edge_deselected:
        return []
    if umesh.is_full_edge_selected:
        return umesh.bm.edges
    return [e for e in umesh.bm.edges if e.select]


def calc_visible_uv_faces_iter(umesh: 'utypes.UMesh') -> typing.Iterable[BMFace]:
    if umesh.is_full_face_selected:
        return umesh.bm.faces
    if umesh.sync:
        return (f for f in umesh.bm.faces if not f.hide)

    if umesh.is_full_face_deselected:
        return []
    return (f for f in umesh.bm.faces if f.select)


def calc_visible_uv_faces(umesh) -> typing.Iterable[BMFace]:
    if umesh.is_full_face_selected:
        return umesh.bm.faces
    if umesh.sync:
        return [f for f in umesh.bm.faces if not f.hide]

    if umesh.is_full_face_deselected:
        return []
    return [f for f in umesh.bm.faces if f.select]

if USE_GENERIC_UV_SYNC:
    def calc_unselected_uv_faces_iter(umesh: 'utypes.UMesh') -> typing.Iterable[BMFace]:
        if umesh.sync:
            if not umesh.sync_valid:
                if umesh.is_full_face_selected:
                    return []
                return (f for f in umesh.bm.faces if not (f.select or f.hide))
            if umesh.is_full_face_selected:
                return (f for f in umesh.bm.faces if not f.uv_select)
            return (f for f in umesh.bm.faces if not (f.uv_select or f.hide))
        else:
            if umesh.is_full_face_deselected:
                return []
            if umesh.is_full_face_selected:
                return (f for f in umesh.bm.faces if not f.uv_select)
            return (f for f in umesh.bm.faces if not f.uv_select and f.select)
else:
    def calc_unselected_uv_faces_iter(umesh: 'utypes.UMesh') -> typing.Iterable[BMFace]:
        if umesh.sync:
            if umesh.is_full_face_selected:
                return []
            return (f for f in umesh.bm.faces if not (f.select or f.hide))
        else:
            if umesh.is_full_face_deselected:
                return []
            uv = umesh.uv
            if umesh.elem_mode == 'EDGE':
                return (f for f in umesh.bm.faces if f.select and not all(crn[uv].select_edge for crn in f.loops))
            return (f for f in umesh.bm.faces if f.select and not all(crn[uv].select for crn in f.loops))


def calc_unselected_uv_faces(umesh: 'utypes.UMesh') -> list[BMFace]:
    return list(calc_unselected_uv_faces_iter(umesh))


def calc_uv_faces(umesh: 'utypes.UMesh', *, selected) -> typing.Iterable[BMFace]:
    if selected:
        return calc_selected_uv_faces(umesh)
    return calc_visible_uv_faces(umesh)


if USE_GENERIC_UV_SYNC:
    def calc_selected_uv_vert_corners(umesh: 'utypes.UMesh') -> list[BMLoop]:
        if umesh.is_full_vert_deselected:
            return []

        if umesh.sync:
            if not umesh.sync_valid:
                if umesh.is_full_vert_selected:
                    return [crn for f in umesh.bm.faces for crn in f.loops]
                return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.vert.select]
            if umesh.is_full_face_selected:
                return [crn for f in umesh.bm.faces for crn in f.loops if crn.uv_select_vert]
            return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.uv_select_vert]

        if umesh.is_full_face_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops if crn.uv_select_vert]
        return [crn for f in umesh.bm.faces if f.select for crn in f.loops if crn.uv_select_vert]
else:
    def calc_selected_uv_vert_corners(umesh: 'utypes.UMesh') -> list[BMLoop]:
        if umesh.is_full_vert_deselected:
            return []

        if umesh.sync:
            if umesh.is_full_vert_selected:
                return [crn for f in umesh.bm.faces for crn in f.loops]
            return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.vert.select]

        uv = umesh.uv
        if umesh.is_full_face_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select]
        return [crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select]

if USE_GENERIC_UV_SYNC:
    def calc_selected_uv_vert_corners_iter(umesh: 'utypes.UMesh') -> 'typing.Generator[BMLoop] | tuple':
        if umesh.is_full_vert_deselected:
            return ()

        if umesh.sync:
            if not umesh.sync_valid:
                if umesh.is_full_vert_selected:
                    return (crn for f in umesh.bm.faces for crn in f.loops)
                return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.vert.select)
            if umesh.is_full_face_selected:
                return (crn for f in umesh.bm.faces for crn in f.loops if crn.uv_select_vert)
            return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.uv_select_vert)

        if umesh.is_full_face_deselected:
            return ()
        if umesh.is_full_face_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops if crn.uv_select_vert)
        return (crn for f in umesh.bm.faces if f.select for crn in f.loops if crn.uv_select_vert)
else:
    def calc_selected_uv_vert_corners_iter(umesh: 'utypes.UMesh') -> 'typing.Generator[BMLoop] | tuple':
        if umesh.sync:
            if umesh.is_full_vert_deselected:
                return ()

            if umesh.is_full_vert_selected:
                return (crn for f in umesh.bm.faces for crn in f.loops)
            return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.vert.select)

        if umesh.is_full_face_deselected:
            return ()

        uv = umesh.uv
        if umesh.is_full_face_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select)
        return (crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select)

if USE_GENERIC_UV_SYNC:
    def calc_selected_uv_edge_corners_iter(umesh: 'utypes.UMesh') -> typing.Iterable[BMLoop]:
        if umesh.is_full_edge_deselected:
            return ()

        if umesh.sync:
            if not umesh.sync_valid:
                if umesh.is_full_edge_selected:
                    return (crn for f in umesh.bm.faces for crn in f.loops)
                return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.edge.select)
            if umesh.is_full_face_selected:
                return (crn for f in umesh.bm.faces for crn in f.loops if crn.uv_select_edge)
            return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.uv_select_edge)

        if umesh.is_full_face_deselected:
            return ()
        if umesh.is_full_face_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops if crn.uv_select_edge)
        return (crn for f in umesh.bm.faces if f.select for crn in f.loops if crn.uv_select_edge)
else:
    def calc_selected_uv_edge_corners_iter(umesh: 'utypes.UMesh') -> typing.Iterable[BMLoop]:
        if umesh.sync:
            if umesh.is_full_edge_deselected:
                return ()

            if umesh.is_full_edge_selected:
                return (crn for f in umesh.bm.faces for crn in f.loops)
            return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.edge.select)

        if umesh.is_full_face_deselected:
            return ()

        uv = umesh.uv
        if umesh.is_full_face_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select_edge)
        return (crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select_edge)

if USE_GENERIC_UV_SYNC:
    def calc_selected_uv_edge_corners(umesh: 'utypes.UMesh') -> list[BMLoop]:
        if umesh.is_full_edge_deselected:
            return []

        if umesh.sync:
            if not umesh.sync_valid:
                if umesh.is_full_edge_selected:
                    return [crn for f in umesh.bm.faces for crn in f.loops]
                return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.edge.select]
            if umesh.is_full_face_selected:
                return [crn for f in umesh.bm.faces for crn in f.loops if crn.uv_select_edge]
            return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.uv_select_edge]

        if umesh.is_full_face_deselected:
            return []
        if umesh.is_full_face_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops if crn.uv_select_edge]
        return [crn for f in umesh.bm.faces if f.select for crn in f.loops if crn.uv_select_edge]
else:
    def calc_selected_uv_edge_corners(umesh: 'utypes.UMesh') -> list[BMLoop]:
        if umesh.sync:
            if umesh.is_full_edge_deselected:
                return []
            if umesh.is_full_face_selected:
                return [crn for f in umesh.bm.faces for crn in f.loops]
            return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops if crn.edge.select]

        if umesh.is_full_face_deselected:
            return []
        uv = umesh.uv
        if umesh.is_full_face_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops if crn[uv].select_edge]
        return [crn for f in umesh.bm.faces if f.select for crn in f.loops if crn[uv].select_edge]

def calc_visible_uv_corners(umesh: 'utypes.UMesh') -> list[BMLoop]:
    if umesh.sync:
        if umesh.is_full_face_selected:
            return [crn for f in umesh.bm.faces for crn in f.loops]
        return [crn for f in umesh.bm.faces if not f.hide for crn in f.loops]

    if umesh.is_full_face_deselected:
        return []
    if umesh.is_full_face_selected:
        return [crn for f in umesh.bm.faces for crn in f.loops]
    return [crn for f in umesh.bm.faces if f.select for crn in f.loops]

# TODO: add calc_unselected_uv_edges func
def calc_visible_uv_corners_iter(umesh: 'utypes.UMesh') -> typing.Iterable[BMLoop]:
    if umesh.sync:
        if umesh.is_full_face_selected:
            return (crn for f in umesh.bm.faces for crn in f.loops)
        return (crn for f in umesh.bm.faces if not f.hide for crn in f.loops)

    if umesh.is_full_face_deselected:
        return []
    if umesh.is_full_face_selected:
        return (crn for f in umesh.bm.faces for crn in f.loops)
    return (crn for f in umesh.bm.faces if f.select for crn in f.loops)


def calc_uv_corners(umesh: 'utypes.UMesh', *, selected) -> list[BMLoop]:
    if selected:
        return calc_selected_uv_vert_corners(umesh)
    return calc_visible_uv_corners(umesh)