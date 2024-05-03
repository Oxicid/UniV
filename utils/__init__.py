import bpy
import math
import bmesh
from collections import defaultdict


class UBMesh:
    def __init__(self, bm, obj, is_edit_bm=True):
        self.bm = bm
        self.obj = obj
        self.is_edit_bm = is_edit_bm

    def update(self, force=False):
        if self.is_edit_bm:
            bmesh.update_edit_mesh(self.obj.data, loop_triangles=force, destructive=force)
        else:
            self.bm.to_mesh(self.obj.data)

    def free(self):
        self.bm.free()

    def __del__(self):
        if not self.is_edit_bm:
            self.bm.free()


class UBMeshSeq:
    def __init__(self, bmeshes):
        self.bmeshes = bmeshes

    @classmethod
    def sel_ob_with_uv(cls):
        bmeshes = []
        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.objects_in_mode_unique_data:
                if obj.type == 'MESH' and obj.data.uv_layers:
                    bm = bmesh.from_edit_mesh(obj.data)
                    bmeshes.append(UBMesh(bm, obj))
        else:
            data_and_objects: defaultdict[bpy.types.Mesh | list[bpy.types.Object]] = defaultdict(list)

            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH' and obj.data.uv_layers:
                    data_and_objects[obj.data].append(obj)

            for data, obj in data_and_objects.items():
                bm = bmesh.new()
                bm.from_mesh(data)
                bmeshes.append(UBMesh(bm, obj, False))

        return cls(bmeshes)

    def __iter__(self):
        return iter(self.bmeshes)

    def __len__(self):
        return len(self.bmeshes)


def find_min_rotate_angle(angle):
    return -(round(angle / (math.pi / 2)) * (math.pi / 2) - angle)


def calc_min_align_angle(selected_faces, uv_layers):
    points = [l[uv_layers].uv for f in selected_faces for l in f.loops]
    align_angle_pre = mathutils.geometry.box_fit_2d(points)
    return find_min_rotate_angle(align_angle_pre)


def calc_min_align_angle_pt(points):
    align_angle_pre = mathutils.geometry.box_fit_2d(points)
    return find_min_rotate_angle(align_angle_pre)
