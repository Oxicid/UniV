# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

from math import pi
from . import transform
from .. import utils
from .. import types
from ..types import BBox, MeshIsland, MeshIslands
from mathutils import Vector, Euler, Matrix

class UNIV_Normal(bpy.types.Operator):
    bl_idname = "mesh.univ_normal"
    bl_label = "Normal"
    bl_description = "Projection by faces normal.\n\nShift - Individual"
    bl_options = {'REGISTER', 'UNDO'}

    crop: bpy.props.BoolProperty(name='Crop', default=True,
                                 description='Packs the islands into a base tile, for performance purposes, does so with uncritical inaccuracy')
    orient: bpy.props.BoolProperty(name='Orient 2D', default=True)
    individual: bpy.props.BoolProperty(name='Individual', default=False, description='Individual by Island Meshes')
    mark_seam: bpy.props.BoolProperty(name='Mark Seam', default=True)
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True,
                                               description='Gets Aspect Correct from the active image from the shader node editor')

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.individual = event.shift
        return self.execute(context)

    def __init__(self):
        self.info = 'No found faces for manipulate'
        self.has_selected: bool = True
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.umeshes = types.UMeshes.calc(self.report)
        self.umeshes.set_sync()
        if self.umeshes.is_edit_mode:
            selected, unselected = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected:
                self.umeshes = selected
                self.has_selected = True
            elif unselected:
                self.umeshes = unselected
                self.has_selected = False
            else:
                return selected.update(info=self.info)
        else:
            self.umeshes.ensure(face=True)

        if self.use_correct_aspect:
            for umesh in self.umeshes:
                umesh.aspect = utils.get_aspect_ratio(umesh)

        if self.individual:
            self.xyz_to_uv_individual()
        else:
            self.xyz_to_uv()

        if not self.umeshes.is_edit_mode:
            ret = self.umeshes.update(info=self.info)
            self.umeshes.free()
            bpy.context.area.tag_redraw()
            return ret
        return self.umeshes.update(info=self.info)

    def xyz_to_uv(self):
        vector_nor, islands_of_mesh = self.avg_normal_and_calc_faces()
        rot_mtx_from_normal = self.calc_rot_mtx_from_normal(vector_nor)

        global_bbox = BBox()
        adv_islands_of_mesh = []

        for mesh_islands in islands_of_mesh:
            adv_island = mesh_islands[0].to_adv_island()
            adv_islands_of_mesh.append(adv_island)
            self.project_orient_and_calc_crop_data(adv_island, global_bbox, rot_mtx_from_normal)

        self.crop_islands(adv_islands_of_mesh, global_bbox)

    def xyz_to_uv_individual(self):
        global_bbox = BBox()
        adv_islands_of_mesh = []

        for vector_nor, mesh_islands in self.avg_normal_and_calc_faces_individual():
            adv_island = mesh_islands.to_adv_island()
            adv_islands_of_mesh.append(adv_island)
            rot_mtx_from_normal = self.calc_rot_mtx_from_normal(vector_nor)
            self.project_orient_and_calc_crop_data(adv_island, global_bbox, rot_mtx_from_normal)

        self.crop_islands(adv_islands_of_mesh, global_bbox)

    def project_orient_and_calc_crop_data(self, adv_island, global_bbox, rot_mtx_from_normal):
        aspect = adv_island.umesh.aspect
        uv = adv_island.umesh.uv

        if aspect >= 1.0:
            aspect_mtx = Matrix.Diagonal((1 / aspect, 1, 1))
        else:
            aspect_mtx = Matrix.Diagonal((1, aspect))
        mtx = aspect_mtx.to_4x4() @ rot_mtx_from_normal @ adv_island.umesh.obj.matrix_world

        points = []
        points_append = points.append

        if self.orient or self.crop:
            for f in adv_island:
                for crn in f.loops:
                    uv_co = (mtx @ crn.vert.co).to_2d()
                    crn[uv].uv = uv_co
                    points_append(uv_co)
        else:
            for f in adv_island:
                for crn in f.loops:
                    crn[uv].uv = (mtx @ crn.vert.co).to_2d()
            return

        convex_coords = utils.calc_convex_points(points)
        bbox = BBox.calc_bbox(convex_coords)

        if self.orient:
            angle = -utils.calc_min_align_angle(convex_coords, aspect)
            adv_island.rotate(angle, bbox.center, aspect)
            bbox.rotate_expand(angle)

        global_bbox.union(bbox)

    def crop_islands(self, adv_islands_of_mesh: list[types.AdvIsland], bbox):
        if not self.crop or not adv_islands_of_mesh:
            return

        if self.mark_seam:
            for isl in adv_islands_of_mesh:
                isl.mark_seam()

        pivot = bbox.left_bottom
        for island in adv_islands_of_mesh:
            if island.umesh.aspect != 1.0:
                if island.umesh.aspect < 1:
                    bbox_ = bbox.copy()
                    bbox_.scale(Vector((1, island.umesh.aspect)), pivot=pivot)
                else:
                    bbox_ = bbox.copy()
                    bbox_.scale(Vector((1/island.umesh.aspect, 1)), pivot=pivot)
            else:
                bbox_ = bbox
            transform.UNIV_OT_Crop.crop_ex('XY', bbox_, inplace=False, islands_of_mesh=[island], offset=Vector((0, 0)), padding=0.001, proportional=True)

    def avg_normal_and_calc_faces_individual(self):
        if self.umeshes.is_edit_mode:
            if self.has_selected:
                calc_mesh_isl_obj = MeshIslands.calc_selected
            else:
                calc_mesh_isl_obj = MeshIslands.calc_visible
        else:
            calc_mesh_isl_obj = MeshIslands.calc_all

        for umesh in self.umeshes:
            _, r, s = umesh.obj.matrix_world.decompose()
            mtx = Matrix.LocRotScale(Vector(), r, s)
            for mesh_isl in calc_mesh_isl_obj(umesh):  # noqa
                weight = Vector()
                for f in mesh_isl:
                    weight += f.normal * f.calc_area()
                weight = mtx @ weight
                yield weight, mesh_isl

    def avg_normal_and_calc_faces(self):
        tot_weight = Vector()
        islands_of_mesh: list[MeshIslands] = []
        for umesh in self.umeshes:
            weight = Vector()
            if not self.umeshes.is_edit_mode:
                faces = umesh.bm.faces
            else:
                if self.has_selected:
                    faces = utils.calc_selected_uv_faces(umesh)
                else:
                    faces = utils.calc_visible_uv_faces(umesh)

            for f in faces:
                weight += f.normal * f.calc_area()

            _, r, s = umesh.obj.matrix_world.decompose()
            mtx = Matrix.LocRotScale(Vector(), r, s)
            tot_weight += mtx @ weight

            islands_of_mesh.append(MeshIslands([MeshIsland(faces, umesh)], umesh))

        return tot_weight, islands_of_mesh

    @staticmethod
    def calc_rot_mtx_from_normal(normal):
        vector_z = Vector((0.0, 0.0, 1.0))

        # rotate x
        vector_n = Vector((0.0, normal.y, normal.z))
        theta_x = vector_z.angle(vector_n, 0)
        vector_cross = vector_n.cross(vector_z)
        if vector_cross.x < 0:
            theta_x = -theta_x
        eul = Euler((theta_x, 0, 0))
        normal.rotate(eul)

        # rotate y
        vector_n = Vector((normal.x, 0.0, normal.z))
        theta_y = vector_z.angle(vector_n, 0)
        vector_cross = vector_n.cross(vector_z)
        if vector_cross.y < 0:
            theta_y = -theta_y
        eul.y = theta_y

        return eul.to_matrix().to_4x4()

class UNIV_BoxProject(bpy.types.Operator):
    bl_idname = "mesh.univ_box_project"
    bl_label = "Box"
    bl_description = "Box Projection"
    bl_options = {'REGISTER', 'UNDO'}

    scale: bpy.props.FloatProperty(name='Scale', default=1, soft_min=0.5, soft_max=2)
    scale_individual: bpy.props.FloatVectorProperty(name='Scale Individual', default=(1.0, 1.0, 1.0), soft_min=0.5, soft_max=2)
    rotation: bpy.props.FloatVectorProperty(name='Rotate', subtype='EULER', soft_min=-pi, soft_max=pi)
    move: bpy.props.FloatVectorProperty(name='Move', subtype='XYZ')
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True,
                                               description='Gets Aspect Correct from the active image from the shader node editor')

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def draw(self, context):
        self.layout.prop(self, 'scale', slider=True)
        col = self.layout.column(align=True)
        col.prop(self, 'scale_individual', expand=True, slider=True)
        col.prop(self, 'rotation', expand=True, slider=True)
        col.prop(self, 'move', expand=True)
        col.separator()
        col.prop(self, 'use_correct_aspect', toggle=1)

    def __init__(self):
        self.is_edit_mode: bool = bpy.context.mode == 'EDIT_MESH'
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.umeshes = types.UMeshes.calc(self.report)
        if self.is_edit_mode:
            self.umeshes.filter_selected_faces()
        self.box()
        if not self.is_edit_mode:
            self.umeshes.update('No faces for manipulate')
            self.umeshes.free()
            bpy.context.area.tag_redraw()
            return {'FINISHED'}
        else:
            return self.umeshes.update(info='Not selected face')

    def box(self):
        move = Vector(self.move) * -1
        scale = Vector(self.scale_individual)*self.scale
        for umesh in self.umeshes:
            uv = umesh.uv

            mtx_from_prop_x = Matrix.LocRotScale(move, Euler((self.rotation[0], 0, 0)), scale)
            mtx_from_prop_y = Matrix.LocRotScale(move, Euler((0, self.rotation[1], 0)), scale)
            mtx_from_prop_z = Matrix.LocRotScale(move, Euler((0, 0, self.rotation[2])), scale)

            if (aspect := (utils.get_aspect_ratio(umesh) if self.use_correct_aspect else 1.0)) >= 1.0:
                aspect_x_mtx = Matrix.Diagonal((1, 1/aspect, 1))
                aspect_y_mtx = Matrix.Diagonal((1/aspect, 1, 1))
                aspect_z_mtx = Matrix.Diagonal((1/aspect, 1, 1))
            else:
                aspect_x_mtx = Matrix.Diagonal((1, 1, aspect))
                aspect_y_mtx = Matrix.Diagonal((1, 1, aspect))
                aspect_z_mtx = Matrix.Diagonal((1, aspect, 1))

            mtx_x = aspect_x_mtx.to_4x4() @ umesh.obj.matrix_world @ mtx_from_prop_x
            mtx_y = aspect_y_mtx.to_4x4() @ umesh.obj.matrix_world @ mtx_from_prop_y
            mtx_z = aspect_z_mtx.to_4x4() @ umesh.obj.matrix_world @ mtx_from_prop_z

            _, r, _ = umesh.obj.matrix_world.decompose()
            faces = (f for f in umesh.bm.faces if f.select) if self.is_edit_mode else umesh.bm.faces
            for f in faces:
                n = f.normal.copy()
                n.rotate(r)
                if abs(n.x) >= abs(n.y) and abs(n.x) >= abs(n.z):  # X
                    for crn in f.loops:
                        crn[uv].uv = (mtx_x @ crn.vert.co).yz
                elif abs(n.y) >= abs(n.x) and abs(n.y) >= abs(n.z):  # Y
                    for crn in f.loops:
                        crn[uv].uv = (mtx_y @ crn.vert.co).xz
                else:  # Z
                    for crn in f.loops:
                        crn[uv].uv = (mtx_z @ crn.vert.co).to_2d()
