import bmesh
from mathutils import Vector
from bmesh.types import *

__all__ = ('face_centroid_uv',)

def face_centroid_uv(f: BMFace, uv_layer: BMLayerItem):
    value = Vector((0, 0))
    loops = f.loops
    for l in loops:
        value += l[uv_layer].uv
    return value / len(loops)
