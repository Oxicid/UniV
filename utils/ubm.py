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


def prev_disc(l: BMLoop) -> BMLoop:
    return l.link_loop_prev.link_loop_radial_prev

def linked_crn_uv(first: BMLoop, uv_layer: BMLayerItem):
    linked = []
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if first[uv_layer].uv == bm_iter[uv_layer].uv:
            linked.append(bm_iter)
    return linked

def calc_crn_in_vert_by_tag(first: BMLoop):
    if not first.tag:
        return []
    linked = [first]
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if bm_iter.tag:
            linked.append(bm_iter)
    return linked

def select_linked_crn_uv_vert(first: BMLoop, uv_layer: BMLayerItem):
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        crn_uv_bm_iter = bm_iter[uv_layer]
        if first[uv_layer].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = True

def select_crn_uv_edge(crn: BMLoop, uv_layer):
    link_crn_next = crn.link_loop_next
    select_linked_crn_uv_vert(crn, uv_layer)
    select_linked_crn_uv_vert(link_crn_next, uv_layer)

    crn_uv_a = crn[uv_layer]
    crn_uv_b = link_crn_next[uv_layer]
    crn_uv_a.select = True
    crn_uv_a.select_edge = True
    crn_uv_b.select = True

def deselect_linked_crn_uv_vert(first: BMLoop, uv_layer: BMLayerItem):
    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        crn_uv_bm_iter = bm_iter[uv_layer]
        if first[uv_layer].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = False

def deselect_crn_uv(first: BMLoop, uv: BMLayerItem):
    first[uv].select_edge = False

    bm_iter = first
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.face.select:
            continue
        crn_uv_bm_iter = bm_iter[uv]
        if first[uv].uv == crn_uv_bm_iter.uv:
            if crn_uv_bm_iter.select:
                break
        else:
            first[uv].select = False

    second = first.link_loop_next
    bm_iter = second
    while True:
        if (bm_iter := prev_disc(bm_iter)) == second:
            break
        if not bm_iter.face.select:
            continue
        crn_uv_bm_iter = bm_iter[uv]
        if second[uv].uv == crn_uv_bm_iter.uv:
            if crn_uv_bm_iter.select:
                break
        else:
            second[uv].select = False

def deselect_crn_uv_force(first: BMLoop, uv: BMLayerItem):
    first[uv].select_edge = False

    bm_iter = first
    first[uv].select = False
    while True:
        if (bm_iter := prev_disc(bm_iter)) == first:
            break
        if not bm_iter.face.select:
            continue
        crn_uv_bm_iter = bm_iter[uv]
        if first[uv].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = False

    second = first.link_loop_next
    second[uv].select = False
    bm_iter = second
    while True:
        if (bm_iter := prev_disc(bm_iter)) == second:
            break
        if not bm_iter.face.select:
            continue
        crn_uv_bm_iter = bm_iter[uv]
        if second[uv].uv == crn_uv_bm_iter.uv:
            crn_uv_bm_iter.select = False


# def deselect_crn_uv_extend(first: BMLoop, uv: BMLayerItem):
#     if not first[uv].select_edge:
#         return
#     first[uv].select_edge = False
#
#     shared_crn = first.link_loop_radial_prev
#     if shared_crn != first:
#         if first[uv].uv == shared_crn.link_loop_next[uv].uv and first.link_loop_next[uv].uv == shared_crn[uv].uv:
#             shared_crn[uv].select_edge = False
#             deselect_linked_crn_uv_vert(shared_crn, uv)
#             deselect_linked_crn_uv_vert(first, uv)
#         elif first[uv].uv == shared_crn.link_loop_next[uv].uv:
#             deselect_linked_crn_uv_vert(first, uv)
#         elif first.link_loop_next[uv].uv == shared_crn[uv].uv:
#             deselect_linked_crn_uv_vert(shared_crn, uv)
#
#
#     bm_iter = first
#     while True:
#         if (bm_iter := _prev_disc(bm_iter)) == first:
#             break
#         if not bm_iter.face.select:
#             continue
#         crn_uv_bm_iter = bm_iter[uv]
#         if first[uv].uv == crn_uv_bm_iter.uv:
#             if crn_uv_bm_iter.select:
#                 break
#         else:
#             first[uv].select = False
#
#     second = first.link_loop_next
#     bm_iter = first
#     while True:
#         if (bm_iter := _prev_disc(bm_iter)) == second:
#             break
#         if not bm_iter.face.select:
#             continue
#         crn_uv_bm_iter = bm_iter[uv]
#         if second[uv].uv == crn_uv_bm_iter.uv:
#             if crn_uv_bm_iter.select:
#                 break
#         else:
#             second[uv].select = False

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
