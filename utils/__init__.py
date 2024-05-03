import bpy
import math
import bmesh
import typing
from collections import defaultdict
from ..types import PyBMesh

class UMesh:
    def __init__(self, bm, obj, is_edit_bm=True):
        self.bm: bmesh.types.BMesh = bm
        self.obj: bpy.types.Object = obj
        self.is_edit_bm: bool = is_edit_bm

    def update(self, force=False):
        if self.is_edit_bm:
            bmesh.update_edit_mesh(self.obj.data, loop_triangles=force, destructive=force)
        else:
            self.bm.to_mesh(self.obj.data)

    def free(self):
        self.bm.free()

    def ensure(self, face=True, edge=False, vert=False, force=False):
        if self.is_edit_bm or not force:
            return
        if face:
            self.bm.faces.ensure_lookup_table()
        if edge:
            self.bm.edges.ensure_lookup_table()
        if vert:
            self.bm.verts.ensure_lookup_table()

    def __del__(self):
        if not self.is_edit_bm:
            self.bm.free()


class UMeshes:
    def __init__(self, umeshes):
        self.umeshes = umeshes

    @classmethod
    def sel_ob_with_uv(cls):
        bmeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH' and obj.data.uv_layers:
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH' and obj.data.uv_layers:
                    data_and_objects[obj.data].append(obj)

            for data, obj in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                bmeshes.append(UMesh(bm, obj, False))

        return cls(bmeshes)

    def __iter__(self):
        return iter(self.umeshes)

    def __len__(self):
        return len(self.umeshes)


def selected_uv_faces(bm, uv_layer, sync=bpy.context.scene.tool_settings.use_uv_select_sync) -> typing.Sequence[bmesh.types.BMFace]:
    if PyBMesh.is_full_face_deselected(bm):
        return []

    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return bm.faces
        return [f for f in bm.faces if f.select]

    if PyBMesh.is_full_face_selected(bm):
        return [f for f in bm.faces if all(l[uv_layer].select for l in f.loops)]
    return [f for f in bm.faces if all(l[uv_layer].select for l in f.loops) and f.select]

def selected_uv_faces_iter(bm, uv_layer, sync=bpy.context.scene.tool_settings.use_uv_select_sync) -> typing.Generator[bmesh.types.BMFace] | tuple:
    if PyBMesh.is_full_face_deselected(bm):
        return ()

    if sync:
        if PyBMesh.is_full_face_selected(bm):
            return bm.faces
        return (f for f in bm.faces if f.select)

    if PyBMesh.is_full_face_selected(bm):
        return (f for f in bm.faces if all(l[uv_layer].select for l in f.loops))
    return (f for f in bm.faces if all(l[uv_layer].select for l in f.loops) and f.select)

def selected_uv_corners(bm, uv_layer, sync=bpy.context.scene.tool_settings.use_uv_select_sync) -> typing.Sequence[bmesh.types.BMLoop]:
    if PyBMesh.is_full_face_deselected(bm):
        return []

    if sync:
        if PyBMesh.is_full_vert_selected(bm):
            return [luv for f in bm.faces for luv in f.loops]
        return [luv for f in bm.faces if f.select for luv in f.loops]

    if PyBMesh.is_full_face_selected(bm):
        return [luv for f in bm.faces for luv in f.loops if luv[uv_layer].select]
    return [luv for f in bm.faces if f.select for luv in f.loops if luv[uv_layer].select]


# TODO: Test with different mode
def selected_uv_corners_iter(bm, uv_layer, sync=bpy.context.scene.tool_settings.use_uv_select_sync) -> typing.Generator[bmesh.types.BMLoop] | tuple:
    if sync:
        if PyBMesh.is_full_vert_deselected(bm):
            return ()
        if PyBMesh.is_full_vert_selected(bm):
            return (luv for f in bm.faces for luv in f.loops)
        return (luv for f in bm.faces if not f.hide for luv in f.loops if luv.vert.select)

    if PyBMesh.is_full_face_deselected(bm):
        return ()
    if PyBMesh.is_full_face_selected(bm):
        return (luv for f in bm.faces for luv in f.loops if luv[uv_layer].select)
    return (luv for f in bm.faces if f.select for luv in f.loops if luv[uv_layer].select)


def find_min_rotate_angle(angle):
    return -(round(angle / (math.pi / 2)) * (math.pi / 2) - angle)


def calc_min_align_angle(selected_faces, uv_layers):
    points = [l[uv_layers].uv for f in selected_faces for l in f.loops]
    align_angle_pre = mathutils.geometry.box_fit_2d(points)
    return find_min_rotate_angle(align_angle_pre)


def calc_min_align_angle_pt(points):
    align_angle_pre = mathutils.geometry.box_fit_2d(points)
    return find_min_rotate_angle(align_angle_pre)
