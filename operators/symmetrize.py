# SPDX-FileCopyrightText: 2025 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math

from .. import utils
from .. import utypes
from mathutils import Vector, kdtree

class UNIV_OT_Symmetrize(bpy.types.Operator):
    bl_idname = "uv.univ_symmetrize"
    bl_label = "Symmetrize"
    bl_description = "Symmetrize"
    bl_options = {'REGISTER', 'UNDO'}

    axis_uv: bpy.props.EnumProperty(name='UV Axis', default='-X to +X',
                                      items=(('-X to +X', '-X to +X', ''), ('-Y to +Y', '-Y to +Y', '')))
    axis_uv_flip: bpy.props.BoolProperty(name='Flip UV Axis ', default=False,
                                         description="Flip selection between (does not affect when one face is unselected)")

    axis_3d: bpy.props.EnumProperty(name='3D Axis', default='X', items=(('X', 'X', ''), ('Y', 'Y', ''), ('Z', 'Z', '')))
    by_cursor: bpy.props.BoolProperty(name='By Cursor', default=False)
    unlink: bpy.props.BoolProperty(name='Unlink', default=False)
    threshold: bpy.props.FloatProperty(name='Threshold', default=0.005, soft_min=0.005, min=0.00001, soft_max=1, max=10)

    def draw(self, context):
        row = self.layout.row(align=True, heading='UV Axis')
        row.prop(self, 'axis_uv', expand=True)
        row.separator(factor=0.35)
        row.prop(self, 'axis_uv_flip', icon_only=True, icon='ARROW_LEFTRIGHT')

        row = self.layout.row(align=True, heading='3D Axis')
        row.scale_x=1.3
        row.prop(self, 'axis_3d', expand=True, slider=True)

        self.layout.prop(self, 'unlink')
        self.layout.prop(self, 'by_cursor')
        self.layout.prop(self, 'threshold')

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def execute(self, context):
        umeshes = utypes.UMeshes(report=self.report)
        umeshes.filter_by_selected_uv_faces()

        not_matched = 0
        if self.axis_3d == 'X':
            xyz_mirror_scale = Vector((-1, 1, 1))
        elif self.axis_3d == 'Y':
            xyz_mirror_scale = Vector((1, -1, 1))
        else: # self.axis_3d == 'X':
            xyz_mirror_scale = Vector((1, 1, -1))

        if self.axis_uv == '-X to +X':
            uv_mirror_scale = Vector((-1, 1))
        else:
            uv_mirror_scale = Vector((1, -1))

        for umesh in umeshes:
            uv = umesh.uv
            faces = utils.calc_visible_uv_faces(umesh)
            if not isinstance(faces, list):
                faces = list(faces)

            tree = kdtree.KDTree(len(faces))
            for idx, f, in enumerate(faces):
                tree.insert(f.calc_center_median(), idx)
            tree.balance()

            # Islands are sorted by their bounding box to determine which one is the source and which is the destination.
            islands = utypes.AdvIslands.calc_selected_with_mark_seam(umesh)
            if self.axis_uv == '-X to +X':
                islands.islands.sort(key=lambda isl_: isl_.bbox.center.x % 1, reverse=self.axis_uv_flip)
            else:
                islands.islands.sort(key=lambda isl_: isl_.bbox.center.y % 1, reverse=self.axis_uv_flip)
            islands.indexing()

            for f in faces:
                f.tag = True

            # Faces inner island are sorted along the 3D axes to avoid random determination of the source and destination.
            # For cases where the transferred faces belong to the same part of an island.
            for isl in islands:
                if self.axis_3d == 'X':
                    isl.faces.sort(key=lambda f_: f_.calc_center_median().x)
                elif self.axis_3d == 'Y':
                    isl.faces.sort(key=lambda f_: f_.calc_center_median().y)
                else:  # self.axis_3d == 'X':
                    isl.faces.sort(key=lambda f_: f_.calc_center_median().z)

            # Set pivots
            for isl in islands:
                if self.by_cursor and (loc := utils.get_cursor_location()):
                    isl.value = loc
                else:
                    if self.axis_uv == '-X to +X':
                        pivot_u = math.floor(isl.bbox.center.x)
                        if pivot_u < 0:
                            isl.value = Vector((pivot_u + -math.copysign(0.5, pivot_u), 0))
                        else:
                            isl.value = Vector((pivot_u + math.copysign(0.5, pivot_u), 0))
                    else:
                        pivot_v = math.floor(isl.bbox.center.y)
                        if pivot_v < 0:
                            isl.value = Vector((0, pivot_v + -math.copysign(0.5, pivot_v)))
                        else:
                            isl.value = Vector((0, pivot_v + math.copysign(0.5, pivot_v)))

            has_update = False
            for isl in islands:
                for src_f in isl:
                    if not src_f.tag:
                        continue
                    src_f.tag = False

                    symmetric_3d_point = src_f.calc_center_median() * xyz_mirror_scale
                    for _, idx, _ in tree.find_range(symmetric_3d_point, self.threshold):
                        dst_f = faces[idx]
                        if not dst_f.tag or len(dst_f.verts) != (size := len(dst_f.verts)):
                            continue

                        dst_f.tag = False

                        # Faces on the same islands aren’t flipped, since the sorting is based on the overall bounding box,
                        # which determines which face is the source and which is the destination. That’s why we force a manual swap.
                        if self.axis_uv_flip and dst_f.index == src_f.index:
                            src_f, dst_f = dst_f, src_f

                        f_tree = kdtree.KDTree(size)
                        for idx_, v in enumerate(dst_f.verts):
                            f_tree.insert(v.co, idx_)
                        f_tree.balance()

                        round_pivot = isl.value
                        scale_diff = round_pivot - round_pivot * uv_mirror_scale
                        for crn in src_f.loops:
                            _, target_crn_idx, _ = f_tree.find(crn.vert.co * xyz_mirror_scale)

                            # Transfer uv coords
                            dst_crn = dst_f.loops[target_crn_idx]
                            uv_co = (crn[uv].uv * uv_mirror_scale + scale_diff)
                            if not self.unlink:
                                dst_linked_loops = utils.linked_crn_to_vert_pair(dst_crn, uv, umesh.sync)
                                for l_crn in dst_linked_loops:
                                    l_crn[uv].uv = uv_co
                            dst_crn[uv].uv = uv_co

                        has_update = True
                        break
                    else:
                        not_matched += 1

            umesh.update_tag = bool(has_update)

        if not_matched and umeshes.update_tag:
            self.report({'WARNING'}, f'Symmetrize: {not_matched!r} faces not matched')
        umeshes.update()
        return {'FINISHED'}
