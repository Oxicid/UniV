# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy

from ..types import Islands
from .. import utils
from ..utils import UMeshes

class UnwrapData:
    def __init__(self, umesh, pins, island, selected):
        self.umesh = umesh
        self.pins = pins
        self.islands = island
        self.temp_selected = selected

class UNIV_OT_Unwrap(bpy.types.Operator):
    bl_idname = "uv.univ_unwrap"
    bl_label = "Unwrap"
    bl_options = {'REGISTER', 'UNDO'}

    unwrap: bpy.props.EnumProperty(name='Unwrap', default='ANGLE_BASED', items=(('ANGLE_BASED', 'Angle Based', ''), ('CONFORMAL', 'Conformal', '')))
    axis: bpy.props.EnumProperty(name='Axis', default='BOTH', items=(('BOTH', 'Both', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    blend_factor: bpy.props.FloatProperty(name='Blend Factor', default=1, soft_min=0, soft_max=1)

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def draw(self, context):
        self.layout.row(align=True).prop(self, 'unwrap', expand=True)
        self.layout.row(align=True).prop(self, 'axis', expand=True)
        self.layout.prop(self, 'blend_factor', slider=True)

    def invoke(self, context, event):
        return self.execute(context)

    def __init__(self):
        self.sync: bool = utils.sync()
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        self.umeshes = UMeshes()
        if self.sync:
            if bpy.context.tool_settings.mesh_select_mode[2]:
                self.unwrap_sync_faces()
            else:
                self.unwrap_sync_verts_edges()
        else:
            self.unwrap_non_sync()

        for umesh in self.umeshes:
            umesh.bm.select_flush_mode()
        return self.umeshes.update()

    def unwrap_sync_verts_edges(self):
        pin_and_inplace = []
        unwrap_data: list[UnwrapData] = []
        for umesh in reversed(self.umeshes):
            if bpy.context.tool_settings.mesh_select_mode[1]:  # EDGE
                if umesh.is_full_edge_deselected:
                    self.umeshes.umeshes.remove(umesh)
                    continue
            elif bpy.context.tool_settings.mesh_select_mode[0]:  # VERTEX
                if umesh.is_full_vert_deselected:
                    self.umeshes.umeshes.remove(umesh)
                    continue
            else:
                raise NotImplemented

            uv = umesh.uv_layer
            islands = Islands.calc_extended_any_elem_with_mark_seam(umesh)

            for isl in islands:
                isl.mark_seam(additional=True)

            faces_to_select = set()
            verts_to_select = set()

            # Extend selected
            for f in umesh.bm.faces:
                if f.hide or f.select:
                    continue
                if sum(v.select for v in f.verts) not in (0, len(f.verts)):
                    faces_to_select.add(f)
                    for v in f.verts:
                        if not v.select:
                            verts_to_select.add(v)

            for f in faces_to_select:
                f.select = True

            unpin_uvs = set()
            for v in verts_to_select:
                v.select = True
                for crn in v.link_loops:
                    crn_uv = crn[uv]
                    if not crn_uv.pin_uv:
                        crn_uv.pin_uv = True
                        unpin_uvs.add(crn_uv)

            if bpy.context.tool_settings.mesh_select_mode[1]:  # EDGE
                for e in umesh.bm.edges:
                    e.select = sum(v.select for v in e.verts) == 2

            save_transform_islands = []
            for isl in islands:
                if any(v.select for f in isl for v in f.verts):
                    save_t = isl.save_transform()
                    save_t.save_coords(self.axis, self.blend_factor)
                    save_transform_islands.append(save_t)

            pin_and_inplace.append((unpin_uvs, save_transform_islands))
            unwrap_data.append(UnwrapData(umesh, unpin_uvs, save_transform_islands, verts_to_select))

        bpy.ops.uv.unwrap(method=self.unwrap)

        for ud in unwrap_data:
            for pin in ud.pins:
                pin.pin_uv = False
            for isl in ud.islands:
                isl.inplace()
                isl.apply_saved_coords(self.axis, self.blend_factor)
            for v in ud.temp_selected:
                v.select = False

            if bpy.context.tool_settings.mesh_select_mode[1]:  # EDGE
                for e in ud.umesh.bm.edges:
                    e.select = sum(v.select for v in e.verts) == 2

    def unwrap_sync_faces(self):
        assert bpy.context.tool_settings.mesh_select_mode[2]
        from ..utils import linked_crn_uv, shared_is_linked

        unwrap_data: list = []
        for umesh in reversed(self.umeshes):
            if umesh.is_full_face_deselected:
                self.umeshes.umeshes.remove(umesh)
                continue

            uv = umesh.uv_layer
            islands_extended = Islands.calc_extended_with_mark_seam(umesh)

            selected_islands = Islands.calc_selected_with_mark_seam(umesh)
            selected_islands.indexing(force=True)

            save_transform_islands = []
            for isl in islands_extended:
                isl.mark_seam(additional=True)
                save_t = isl.save_transform()
                save_t.save_coords(self.axis, self.blend_factor)
                save_transform_islands.append(save_t)

            pinned = []
            to_select = []
            if not umesh.is_full_face_selected:
                for f in selected_islands.faces_iter():
                    for crn in f.loops:
                        for linked_crn in linked_crn_uv(crn, uv):
                            linked_crn_face = linked_crn.face

                            if linked_crn_face.index != -1 or linked_crn_face.hide:
                                continue

                            linked_crn_face.index = f.index
                            to_select.append(linked_crn_face)

                for f in to_select:
                    f_index = f.index
                    for crn in f.loops:
                        if (shared_crn := crn.link_loop_radial_prev) == crn:
                            crn_uv = crn[uv]
                            if not crn_uv.pin_uv:
                                crn_uv.pin_uv = True
                                pinned.append(crn_uv)
                            continue

                        shared_crn_face = shared_crn.face
                        shared_face_index = shared_crn_face.index
                        if shared_face_index == f_index:
                            continue
                        if shared_crn_face.select or shared_crn_face.hide:
                            continue
                        if not shared_is_linked(crn, shared_crn, uv):
                            continue

                        shared_crn_uv = crn[uv]
                        if not shared_crn_uv.pin_uv:
                            shared_crn_uv.pin_uv = True
                            pinned.append(shared_crn_uv)

                        shared_crn_next_uv = crn.link_loop_next[uv]
                        if not shared_crn_next_uv.pin_uv:
                            shared_crn_next_uv.pin_uv = True
                            pinned.append(shared_crn_next_uv)

                for f in to_select:
                    f.select = True
            unwrap_data.append((pinned, to_select, save_transform_islands))

        bpy.ops.uv.unwrap(method=self.unwrap)

        for pinned, faces, islands in unwrap_data:
            for pin in pinned:
                pin.pin_uv = False
            for f in faces:
                f.select = False
            for isl in islands:
                isl.inplace()
                isl.apply_saved_coords(self.axis, self.blend_factor)

    def unwrap_non_sync(self):
        save_transform_islands = []
        for umesh in reversed(self.umeshes):
            uv = umesh.uv_layer
            if umesh.is_full_face_deselected or not any(crn[uv].select for f in umesh.bm.faces if f.select for crn in f.loops):
                self.umeshes.umeshes.remove(umesh)
                continue

            islands = Islands.calc_extended_any_elem_with_mark_seam(umesh)

            for isl in islands:
                isl.mark_seam(additional=True)

            for isl in islands:
                save_t = isl.save_transform()
                save_t.save_coords(self.axis, self.blend_factor)
                save_transform_islands.append(save_t)

        bpy.ops.uv.unwrap(method=self.unwrap)

        for isl in save_transform_islands:
            isl.inplace()
            isl.apply_saved_coords(self.axis, self.blend_factor)