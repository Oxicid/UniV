# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import math

from .transform import UNIV_OT_Crop
from .. import utils
from ..types import AdvIslands, AdvIsland, BBox
from bmesh.types import BMFace
from mathutils import Vector, Euler, Matrix

class UNIV_NProject(bpy.types.Operator):
    bl_idname = "mesh.univ_n_project"
    bl_label = "Project by Normal"
    bl_description = "Projection by faces normal"
    bl_options = {'REGISTER', 'UNDO'}

    crop: bpy.props.BoolProperty(name='Crop', default=True)
    orient: bpy.props.BoolProperty(name='Orient', default=True)

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def __init__(self):
        self.umeshes: utils.UMeshes | None = None
        self.islands_of_mesh: list[AdvIslands] = []  # TODO: Replace to FaceIsland after refactor

    def execute(self, context):
        self.umeshes = utils.UMeshes.calc()
        self.umeshes.filter_selected_faces()
        self.islands_of_mesh = []
        self.xyz_to_uv()
        return self.umeshes.update(info='Not selected face')

    def xyz_to_uv(self):
        crop = self.crop
        vector_nor = self.avg_normal_and_calc_selected_faces()
        rot_mtx_from_normal = self.calc_rot_mtx_from_normal(vector_nor)

        xmin = math.inf
        xmax = -math.inf
        ymin = math.inf
        ymax = -math.inf

        for island, umesh in zip(self.islands_of_mesh, self.umeshes):
            uv = island.uv_layer
            mtx = rot_mtx_from_normal @ umesh.obj.matrix_world
            for f in island[0]:
                for crn in f.loops:
                    uv_co = (mtx @ crn.vert.co).to_2d()
                    crn[uv].uv = uv_co
                    if crop:
                        x, y = uv_co
                        if xmin > x:
                            xmin = x
                        if xmax < x:
                            xmax = x
                        if ymin > y:
                            ymin = y
                        if ymax < y:
                            ymax = y

        bbox = BBox(xmin, xmax, ymin, ymax)

        if self.orient:
            points = []
            for island in self.islands_of_mesh:
                if not points:
                    points = island[0].calc_convex_points()
                else:
                    points.extend(island[0].calc_convex_points())
            angle = utils.calc_min_align_angle(points)

            for island in self.islands_of_mesh:
                island.rotate_simple(angle)
            bbox = BBox.calc_bbox(points)

        if crop:
            UNIV_OT_Crop.crop_ex('XY', bbox, inplace=False, islands_of_mesh=self.islands_of_mesh, offset=Vector((0, 0)), padding=0.001, proportional=True)

    def avg_normal_and_calc_selected_faces(self):
        tot_weight = Vector()
        for umesh in self.umeshes:
            weight = Vector()
            selected_faces: list[BMFace] = []
            selected_faces_append = selected_faces.append

            selected_faces_iter = (f for f in umesh.bm.faces if f.select)
            for f, _ in zip(selected_faces_iter, range(umesh.total_face_sel)):
                selected_faces_append(f)
                weight += f.normal * f.calc_area()

            _, r, s = umesh.obj.matrix_world.decompose()
            mtx = Matrix.LocRotScale(Vector(), r, s)
            tot_weight += mtx @ weight

            self.islands_of_mesh.append(AdvIslands([AdvIsland(selected_faces, umesh.bm, umesh.uv_layer)], umesh.bm, umesh.uv_layer))

        return tot_weight

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
