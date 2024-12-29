# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
from .. import types

class UnwrapData:
    def __init__(self, umesh, pins, island, selected):
        self.umesh = umesh
        self.pins = pins
        self.islands = island
        self.temp_selected = selected


items = [('ANGLE_BASED', 'Hard Surface', ''), ('CONFORMAL', 'Conformal', '')]
_bl_description = 'Inplace unwrap the mesh of object being edited'
if bpy.app.version >= (4, 3, 0):
    items.append(('MINIMUM_STRETCH', 'Organic', ''))
    _bl_description += "\n\nOrganic Mode has incorrect behavior with pinned and flipped islands"

MULTIPLAYER = 1
UNIQUE_NUMBER_FOR_MULTIPLY = -1

class UNIV_OT_Unwrap(bpy.types.Operator):
    bl_idname = "uv.univ_unwrap"
    bl_label = "Unwrap"
    bl_description = _bl_description
    bl_options = {'REGISTER', 'UNDO'}

    unwrap: bpy.props.EnumProperty(name='Unwrap', default='ANGLE_BASED', items=items)
    unwrap_along: bpy.props.EnumProperty(name='Unwrap Along', default='BOTH', items=(('BOTH', 'Both', ''), ('X', 'U', ''), ('Y', 'V', '')),
                description="Doesnt work properly with disk-shaped topologies, which completely change their structure with default unwrap")
    blend_factor: bpy.props.FloatProperty(name='Blend Factor', default=1, soft_min=0, soft_max=1)
    mark_seam_inner_island: bpy.props.BoolProperty(name='Mark Seam Inner Island', default=True, description='Marks seams where there are split edges')

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):

        self.layout.prop(self, 'mark_seam_inner_island')

        # col = self.layout.column(align=True)
        col = self.layout.column(align=False)
        split = col.split(factor=0.3, align=True)
        split.label(text='Unwrap Along')
        row = split.row(align=True)
        row.prop(self, 'unwrap_along', expand=True)

        self.layout.prop(self, 'blend_factor', slider=True)
        self.layout.row(align=True).prop(self, 'unwrap', expand=True)

    def invoke(self, context, event):
        return self.execute(context)

    def __init__(self):
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        if context.area.ui_type != 'UV':
            self.umeshes.set_sync(True)

        self.umeshes = types.UMeshes()
        if self.umeshes.sync:
            if bpy.context.tool_settings.mesh_select_mode[2]:
                self.unwrap_sync_faces()
            else:
                self.unwrap_sync_verts_edges()
        else:
            self.unwrap_non_sync()

        for umesh in self.umeshes:
            umesh.bm.select_flush_mode()
        return self.umeshes.update()

    def unwrap_sync_verts_edges(self, **unwrap_kwargs):
        unique_number_for_multiply = 0
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

            uv = umesh.uv
            islands = types.Islands.calc_extended_any_elem_with_mark_seam(umesh)

            if not self.mark_seam_inner_island:
                islands.indexing()

            for isl in islands:
                if unwrap_kwargs:
                    unique_number_for_multiply += hash(isl[0])  # multiplayer
                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)

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
                    save_t.save_coords(self.unwrap_along, self.blend_factor)
                    save_transform_islands.append(save_t)

            pin_and_inplace.append((unpin_uvs, save_transform_islands))
            unwrap_data.append(UnwrapData(umesh, unpin_uvs, save_transform_islands, verts_to_select))

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)
        bpy.ops.uv.unwrap(method=self.unwrap, **unwrap_kwargs)

        for ud in unwrap_data:
            for pin in ud.pins:
                pin.pin_uv = False
            for isl in ud.islands:
                isl.inplace(self.unwrap_along)
                isl.apply_saved_coords(self.unwrap_along, self.blend_factor)
            for v in ud.temp_selected:
                v.select = False

            if bpy.context.tool_settings.mesh_select_mode[1]:  # EDGE
                for e in ud.umesh.bm.edges:
                    e.select = sum(v.select for v in e.verts) == 2

    def unwrap_sync_faces(self, **unwrap_kwargs):
        assert bpy.context.tool_settings.mesh_select_mode[2]
        from ..utils import linked_crn_uv_by_island_index_unordered, \
            linked_crn_uv_by_island_index_unordered_included
        unique_number_for_multiply = 0

        unwrap_data: list = []
        for umesh in reversed(self.umeshes):
            if umesh.is_full_face_deselected:
                self.umeshes.umeshes.remove(umesh)
                continue

            uv = umesh.uv
            islands_extended = types.Islands.calc_extended_with_mark_seam(umesh)
            islands_extended.indexing()

            save_transform_islands = []
            for isl in islands_extended:
                if unwrap_kwargs:
                    unique_number_for_multiply += hash(isl[0])  # multiplayer

                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)
                save_t = isl.save_transform()
                save_t.save_coords(self.unwrap_along, self.blend_factor)
                save_transform_islands.append(save_t)

            if umesh.has_full_selected_uv_faces:
                unwrap_data.append(([], [], save_transform_islands))
                continue

            pinned = []
            to_select_groups = []
            for idx, island in enumerate(islands_extended):
                if island.is_full_face_selected:
                    continue
                to_select = []
                for f in island:
                    if not f.select:
                        continue
                    for crn in f.loops:
                        for linked_crn in linked_crn_uv_by_island_index_unordered(crn, uv, idx):
                            linked_crn_face = linked_crn.face
                            if linked_crn_face.select or linked_crn_face.tag:
                                continue
                            linked_crn_face.tag = True
                            # add neighboring non-selected faces
                            to_select.append(linked_crn_face)
                to_select_groups.append(to_select)

                island.set_corners_tag(False)
                for f in to_select:
                    for crn in f.loops:
                        if crn.tag:
                            continue
                        linked_corners = linked_crn_uv_by_island_index_unordered_included(crn, uv, idx)
                        for crn_ in linked_corners:
                            crn_.tag = True

                        if any(linked_crn.face.select for linked_crn in linked_corners):
                            continue

                        # if linked corners hasn't selected -> set pin
                        for crn_ in linked_corners:
                            crn_uv = crn_[uv]
                            if not crn_uv.pin_uv:
                                crn_uv.pin_uv = True
                                pinned.append(crn_uv)
                            continue

                for f in to_select:
                    f.select = True
            unwrap_data.append((pinned, to_select_groups, save_transform_islands))

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)
        bpy.ops.uv.unwrap(method=self.unwrap, **unwrap_kwargs)

        for pinned, faces_groups, islands in unwrap_data:
            for pin in pinned:
                pin.pin_uv = False
            for faces in faces_groups:
                for f in faces:
                    f.select = False
            for isl in islands:
                isl.inplace(self.unwrap_along)
                isl.apply_saved_coords(self.unwrap_along, self.blend_factor)

    def unwrap_non_sync(self, **unwrap_kwargs):
        save_transform_islands = []
        unique_number_for_multiply = 0
        for umesh in reversed(self.umeshes):
            uv = umesh.uv
            if umesh.is_full_face_deselected or not any(crn[uv].select for f in umesh.bm.faces if f.select for crn in f.loops):
                self.umeshes.umeshes.remove(umesh)
                continue

            islands = types.Islands.calc_extended_any_elem_with_mark_seam(umesh)
            if not self.mark_seam_inner_island:
                islands.indexing()

            for isl in islands:
                if unwrap_kwargs:
                    unique_number_for_multiply += hash(isl[0])  # multiplayer

                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)

            for isl in islands:
                save_t = isl.save_transform()
                save_t.save_coords(self.unwrap_along, self.blend_factor)
                save_transform_islands.append(save_t)

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

        bpy.ops.uv.unwrap(method=self.unwrap, **unwrap_kwargs)

        for isl in save_transform_islands:
            isl.inplace(self.unwrap_along)
            isl.apply_saved_coords(self.unwrap_along, self.blend_factor)

    @staticmethod
    def multiply_relax(unique_number_for_multiply, unwrap_kwargs):
        if unwrap_kwargs:
            global MULTIPLAYER
            global UNIQUE_NUMBER_FOR_MULTIPLY
            if UNIQUE_NUMBER_FOR_MULTIPLY == unique_number_for_multiply:
                MULTIPLAYER += 1
                unwrap_kwargs['iterations'] *= MULTIPLAYER
            else:
                MULTIPLAYER = 1
                UNIQUE_NUMBER_FOR_MULTIPLY = unique_number_for_multiply

# class UNIV_OT_Unwrap_VIEW3D(UNIV_OT_Unwrap):
#     bl_idname = "mesh.univ_unwrap"
#     bl_label = "Unwrap"
#     bl_options = {'REGISTER', 'UNDO'}
#
#     blend_factor: bpy.props.FloatProperty(name='Blend Factor', default=1, soft_min=0, soft_max=1)
#     mark_seam_type: bpy.props.EnumProperty(name='Mark Seam Border Type', default='UV_BORDER',
#                                       items=(('UV_BORDER', 'Angle Based', ''), ('SELECTED_BORDER', 'Selected Border', '')))
#
#     def draw(self, context):
#         self.layout.row(align=True).prop(self, 'unwrap', expand=True)
#         self.layout.prop(self, 'mark_seam_inner_island')
