import bmesh
from ..types import PyBMesh
from bmesh.types import *
from mathutils import Vector

def set_faces_tag(faces, tag=True):
    for f in faces:
        f.tag = tag

def face_centroid_uv(f: BMFace, uv_layer: BMLayerItem):
    value = Vector((0, 0))
    loops = f.loops
    for l in loops:
        value += l[uv_layer].uv
    return value / len(loops)


# Need implement disc_next disc_prev
# def calc_non_manifolds_uv(bm, uv_layer):
#     for f in bm.faces:
#         for l in f.loops:  # Running through all the neighboring faces
#             link_face = l.link_loop_radial_next.face
#             for ll in link_face.loops:
#
#                 if ll[uv_layer].uv != l[uv_layer].uv:
#                     continue
#                 # Skip manifold
#                 if (l.link_loop_next[uv_layer].uv == ll.link_loop_prev[uv_layer].uv) or \
#                         (ll.link_loop_next[uv_layer].uv == l.link_loop_prev[uv_layer].uv):
#                     continue
#                 else:
#                     l[uv_layer].select = True
#                     ll[uv_layer].select = True


def calc_non_manifolds(bm: BMesh) -> tuple[set[BMVert], set[BMEdge]]:
    non_manifold_verts = set()
    for v in bm.verts:
        if v.hide:
            continue
        if not v.is_manifold:
            non_manifold_verts.add(v)

    non_manifold_edges = set()
    for f in bm.faces:
        if f.hide:
            continue
        for l in f.loops:
            link_face = l.link_loop_radial_next.face
            if link_face.hide:
                continue
            for ll in link_face.loops:
                if ll.vert != l.vert:
                    continue
                # Skip manifold
                if (l.link_loop_next.vert == ll.link_loop_prev.vert) or \
                        (ll.link_loop_next.vert == l.link_loop_prev.vert):
                    continue
                else:
                    if not l.edge.is_boundary:
                        non_manifold_edges.add(l.edge)
                    if not ll.edge.is_boundary:
                        non_manifold_edges.add(ll.edge)
    return non_manifold_verts, non_manifold_edges


def _prev_disc(l: BMLoop) -> BMLoop:
    return l.link_loop_prev.link_loop_radial_prev

def linked_crn_uv(first: BMLoop, uv_layer: BMLayerItem):
    linked = []
    bm_iter = first
    while True:
        if (bm_iter := _prev_disc(bm_iter)) == first:
            break
        if first[uv_layer].uv == bm_iter[uv_layer].uv:
            linked.append(bm_iter)
    return linked

def select_linked_crn_uv_vert(first: BMLoop, uv_layer: BMLayerItem):
    bm_iter = first
    while True:
        if (bm_iter := _prev_disc(bm_iter)) == first:
            break
        crn_uv_bm_iter = bm_iter[uv_layer]
        if first[uv_layer].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = True

def deselect_linked_crn_uv_vert(first: BMLoop, uv_layer: BMLayerItem):
    bm_iter = first
    while True:
        if (bm_iter := _prev_disc(bm_iter)) == first:
            break
        crn_uv_bm_iter = bm_iter[uv_layer]
        if first[uv_layer].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = False
def is_boundary(crn: BMLoop, uv_layer: BMLayerItem):
    # assert(not l.face.select)

    # We get a clockwise corner, but linked to the end of the current corner
    if (next_linked_disc := crn.link_loop_radial_prev) == crn:
        return True
    if not next_linked_disc.face.select:
        return True
    return not (crn[uv_layer].uv == next_linked_disc.link_loop_next[uv_layer].uv and
                crn.link_loop_next[uv_layer].uv == next_linked_disc[uv_layer].uv)

def is_boundary_sync(crn: BMLoop, uv_layer: BMLayerItem):
    # assert(not l.face.hide)
    if (next_linked_disc := crn.link_loop_radial_prev) == crn:
        return True
    if next_linked_disc.face.hide:
        return True
    return not (crn[uv_layer].uv == next_linked_disc.link_loop_next[uv_layer].uv and
                crn.link_loop_next[uv_layer].uv == next_linked_disc[uv_layer].uv)

def calc_selected_uv_faces(bm, uv_layer, sync) -> list[bmesh.types.BMFace]:
    if PyBMesh.is_full_face_deselected(bm):
        return []

    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return bm.faces
        return [f for f in bm.faces if f.select]

    if PyBMesh.is_full_face_selected(bm):
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return [f for f in bm.faces if all(l[uv_layer].select for l in f.loops)]
        else:
            return [f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops)]
    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return [f for f in bm.faces if all(l[uv_layer].select for l in f.loops) and f.select]
    else:
        return [f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops) and f.select]

def calc_selected_uv_faces_iter(bm, uv_layer, sync) -> 'typing.Generator[bmesh.types.BMFace] | tuple':
    if PyBMesh.is_full_face_deselected(bm):
        return ()

    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return bm.faces
        return (f for f in bm.faces if f.select)

    if PyBMesh.is_full_face_selected(bm):
        if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
            return (f for f in bm.faces if all(l[uv_layer].select for l in f.loops))
        else:
            return (f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops))
    if bpy.context.scene.tool_settings.uv_select_mode == 'VERTEX':
        return (f for f in bm.faces if all(l[uv_layer].select for l in f.loops) and f.select)
    else:
        return (f for f in bm.faces if all(l[uv_layer].select_edge for l in f.loops) and f.select)

def calc_visible_uv_faces(bm, uv_layer, sync) -> list[bmesh.types.BMFace]:  # noqa
    if PyBMesh.is_full_face_selected(bm):
        return bm.faces
    if sync:
        return [f for f in bm.faces if not f.hide]
    return [f for f in bm.faces if f.select]

def calc_uv_faces(bm, uv_layer, sync, *, selected) -> list[bmesh.types.BMFace]:
    if selected:
        return calc_selected_uv_faces(bm, uv_layer, sync)
    return calc_visible_uv_faces(bm, uv_layer, sync)

def calc_selected_uv_corners(bm, uv_layer, sync) -> list[bmesh.types.BMLoop]:
    if PyBMesh.is_full_vert_deselected(bm):
        return []

    if sync:
        if PyBMesh.is_full_vert_selected(bm):
            return [l for f in bm.faces for l in f.loops]
        return [l for f in bm.faces for l in f.loops if l.vert.select]

    if PyBMesh.is_full_face_selected(bm):
        return [l for f in bm.faces for l in f.loops if l[uv_layer].select]
    return [l for f in bm.faces if f.select for l in f.loops if l[uv_layer].select]

def calc_selected_uv_corners_iter(bm, uv_layer, sync) -> 'typing.Generator[bmesh.types.BMLoop] | tuple':
    if PyBMesh.is_full_vert_deselected(bm):
        return ()

    if sync:
        if PyBMesh.is_full_vert_selected(bm):
            return (l for f in bm.faces for l in f.loops)
        return (l for f in bm.faces for l in f.loops if l.vert.select)

    if PyBMesh.is_full_face_selected(bm):
        return (luv for f in bm.faces for luv in f.loops if luv[uv_layer].select)
    return (luv for f in bm.faces if f.select for luv in f.loops if luv[uv_layer].select)

def calc_visible_uv_corners(bm, sync) -> list[bmesh.types.BMLoop]:
    if sync:
        return [luv for f in bm.faces if not f.hide for luv in f.loops]
    if PyBMesh.fields(bm).totfacesel == 0:
        return []
    return [luv for f in bm.faces if (f.select and not f.hide) for luv in f.loops]

def calc_uv_corners(bm, uv_layer, sync, *, selected) -> list[bmesh.types.BMLoop]:
    if selected:
        return calc_selected_uv_corners(bm, uv_layer, sync)
    return calc_visible_uv_corners(bm, sync)
