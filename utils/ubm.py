# import bmesh
from bmesh.types import *
from mathutils import Vector

__all__ = ('face_centroid_uv', 'calc_non_manifolds')

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
