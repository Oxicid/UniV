# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

from math import pi
from . import transform  # noqa: F401 # pylint:disable=unused-import
from .transform import UNIV_OT_Crop
from .. import utils
from .. import types  # noqa: F401 # pylint:disable=unused-import
from ..types import BBox, MeshIsland, MeshIslands
from mathutils import Vector, Euler, Matrix

class UNIV_Normal(bpy.types.Operator):
    bl_idname = "mesh.univ_normal"
    bl_label = "Project by Normal"
    bl_description = "Projection by faces normal"
    bl_options = {'REGISTER', 'UNDO'}

    crop: bpy.props.BoolProperty(name='Crop', default=True)
    orient: bpy.props.BoolProperty(name='Orient', default=True)
    individual: bpy.props.BoolProperty(name='Individual', default=False, description='Individual by Island Meshes')

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.individual = event.alt
        return self.execute(context)

    def __init__(self):
        self.is_obj_mode: bool = bpy.context.mode == 'OBJECT'
        self.umeshes: utils.UMeshes | None = None

    def execute(self, context):
        self.umeshes = utils.UMeshes.calc(self.report)
        if not self.is_obj_mode:
            self.umeshes.filter_selected_faces()
            self.umeshes.set_sync(True)
        else:
            self.umeshes.ensure(face=True)

        if self.individual:
            self.xyz_to_uv_individual()
        else:
            self.xyz_to_uv()

        if self.is_obj_mode:
            self.umeshes.update('No found faces for manipulate')
            self.umeshes.free()
            bpy.context.area.tag_redraw()
            return {'FINISHED'}
        return self.umeshes.update(info='Not selected face')

    def xyz_to_uv(self):
        vector_nor, islands_of_mesh = self.avg_normal_and_calc_faces()
        rot_mtx_from_normal = self.calc_rot_mtx_from_normal(vector_nor)

        points = []
        points_append = points.append
        for island in islands_of_mesh:
            uv = island.umesh.uv_layer
            mtx = rot_mtx_from_normal @ island.umesh.obj.matrix_world
            for f in island[0]:
                for crn in f.loops:
                    uv_co = (mtx @ crn.vert.co).to_2d()
                    crn[uv].uv = uv_co
                    points_append(uv_co)

        if not (self.orient or self.crop):
            return

        uv_islands_of_mesh = [island.to_adv_islands() for island in islands_of_mesh]

        if self.orient:
            angle = utils.calc_min_align_angle(points)
            for island in uv_islands_of_mesh:
                island.rotate_simple(angle)

        if self.crop:
            bbox = BBox()
            for isl in uv_islands_of_mesh:
                bbox.union(isl.calc_bbox())

            UNIV_OT_Crop.crop_ex('XY', bbox, inplace=False, islands_of_mesh=uv_islands_of_mesh, offset=Vector((0, 0)), padding=0.001, proportional=True)

    def avg_normal_and_calc_faces(self):
        tot_weight = Vector()
        islands_of_mesh: list[MeshIslands] = []
        for umesh in self.umeshes:
            weight = Vector()
            if self.is_obj_mode:
                faces = umesh.bm.faces
            else:
                faces = utils.calc_selected_uv_faces_b(umesh)

            for f in faces:
                weight += f.normal * f.calc_area()

            _, r, s = umesh.obj.matrix_world.decompose()
            mtx = Matrix.LocRotScale(Vector(), r, s)
            tot_weight += mtx @ weight

            islands_of_mesh.append(MeshIslands([MeshIsland(faces, umesh)], umesh))

        return tot_weight, islands_of_mesh

    def xyz_to_uv_individual(self):
        points = []
        points_append = points.append
        mesh_islands = []
        for vector_nor, mesh_isl in self.avg_normal_and_calc_faces_individual():
            uv = mesh_isl.umesh.uv_layer
            mesh_islands.append(mesh_isl)
            rot_mtx_from_normal = self.calc_rot_mtx_from_normal(vector_nor)
            mtx = rot_mtx_from_normal @ mesh_isl.umesh.obj.matrix_world

            for f in mesh_isl:
                for crn in f.loops:
                    uv_co = (mtx @ crn.vert.co).to_2d()
                    crn[uv].uv = uv_co
                    points_append(uv_co)

        if not (self.orient or self.crop):
            return

        uv_islands_of_mesh = [island.to_adv_island() for island in mesh_islands]

        # if self.mark_seam:  # TODO: Implement Mark Seam after island struct refactoring (umesh)
        #     for isl in uv_islands_of_mesh:
        #         isl.mark_seam()

        if self.orient:
            angle = utils.calc_min_align_angle(points)  # TODO: Optimize, calc convex, rotate and calc bbox
            for island in uv_islands_of_mesh:
                island.rotate_simple(angle)

        if self.crop:
            bbox = BBox()
            for isl in uv_islands_of_mesh:
                bbox.union(isl.calc_bbox())

            UNIV_OT_Crop.crop_ex('XY', bbox, inplace=False, islands_of_mesh=uv_islands_of_mesh, offset=Vector((0, 0)), padding=0.001, proportional=True)

    def avg_normal_and_calc_faces_individual(self):
        calc_mesh_isl_obj = MeshIslands.calc_all if self.is_obj_mode else MeshIslands.calc_selected
        # if self.is_obj_mode:
        #     calc_mesh_isl_obj = MeshIslands.calc_all
        # else:
        #     calc_mesh_isl_obj = MeshIslands.calc_selected

        for umesh in self.umeshes:
            _, r, s = umesh.obj.matrix_world.decompose()
            mtx = Matrix.LocRotScale(Vector(), r, s)
            for mesh_isl in calc_mesh_isl_obj(umesh):  # noqa
                weight = Vector()
                for f in mesh_isl:
                    weight += f.normal * f.calc_area()
                weight = mtx @ weight
                yield weight, mesh_isl

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
    bl_label = "Project by Box"
    bl_description = "Box Projection by faces normal"
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
        self.is_obj_mode: bool = bpy.context.mode == 'OBJECT'
        self.umeshes: utils.UMeshes | None = None

    def execute(self, context):
        self.umeshes = utils.UMeshes.calc(self.report)
        if not self.is_obj_mode:
            self.umeshes.filter_selected_faces()
        self.box()
        if self.is_obj_mode:
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
            uv = umesh.uv_layer

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
            faces = umesh.bm.faces if self.is_obj_mode else (f for f in umesh.bm.faces if f.select)
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
