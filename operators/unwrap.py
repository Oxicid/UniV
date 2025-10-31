# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
from mathutils import Vector
from .. import utypes
from .. import utils
from ..preferences import prefs
from ..utils import linked_crn_uv_by_island_index_unordered_included


class UnwrapData:
    def __init__(self, umesh, pins, island, selected):
        self.umesh: utypes.UMesh = umesh
        self.pins = pins
        self.islands = island
        self.temp_selected = selected


MULTIPLAYER = 1
UNIQUE_NUMBER_FOR_MULTIPLY = -1


class UNIV_OT_Unwrap(bpy.types.Operator):
    bl_idname = "uv.univ_unwrap"
    bl_label = "Unwrap"
    bl_description = ("Inplace unwrap the mesh of object being edited\n\n "
                      "Organic Mode has incorrect behavior with pinned and flipped islands")
    bl_options = {'REGISTER', 'UNDO'}

    unwrap: bpy.props.EnumProperty(name='Unwrap',
                                   default='ANGLE_BASED',
                                   items=(('ANGLE_BASED', 'Hard Surface', ''),
                                          ('CONFORMAL', 'Conformal', ''),
                                          ('MINIMUM_STRETCH', 'Organic', '')))
    unwrap_along: bpy.props.EnumProperty(name='Unwrap Along', default='BOTH', items=(('BOTH', 'Both', ''), ('X', 'U', ''), ('Y', 'V', '')),
                                         description="Doesnt work properly with disk-shaped topologies, which completely change their structure with default unwrap")
    blend_factor: bpy.props.FloatProperty(name='Blend Factor', default=1, soft_min=0, soft_max=1)
    mark_seam_inner_island: bpy.props.BoolProperty(
        name='Mark Seam Self Borders', default=True, description='Marking seams where there are split edges within the same island.')
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):

        self.layout.prop(self, 'use_correct_aspect')
        self.layout.prop(self, 'mark_seam_inner_island')

        col = self.layout.column()
        split = col.split(factor=0.3, align=True)
        split.label(text='Unwrap Along')
        row = split.row(align=True)
        row.prop(self, 'unwrap_along', expand=True)

        self.layout.prop(self, 'blend_factor', slider=True)
        self.layout.row(align=True).prop(self, 'unwrap', expand=True)

    def invoke(self, context, event):
        if self.bl_idname.startswith('UV'):
            if event.value == 'PRESS':
                self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
                self.mouse_pos = utils.get_mouse_pos(context, event)
            else:
                self.max_distance = None
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_pos = Vector((0, 0))
        self.max_distance: float | None = None
        self.umeshes: utypes.UMeshes | None = None

    def execute(self, context):
        self.umeshes = utypes.UMeshes()
        self.umeshes.fix_context()
        if self.unwrap == 'MINIMUM_STRETCH' and bpy.app.version < (4, 3, 0):
            self.unwrap = 'ANGLE_BASED'
            self.report({'WARNING'}, 'Organic Mode is not supported in Blender versions below 4.3')

        selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_verts()
        self.umeshes = selected_umeshes if selected_umeshes else unselected_umeshes
        if not self.umeshes:
            return self.umeshes.update()

        if not selected_umeshes and self.max_distance is not None and context.area.ui_type == 'UV':
            return self.pick_unwrap()
        else:
            if self.umeshes.sync:
                if self.umeshes.elem_mode == 'FACE':
                    self.unwrap_sync_faces()
                else:
                    self.unwrap_sync_verts_edges()
            else:
                self.unwrap_non_sync()

            for umesh in self.umeshes:
                umesh.bm.select_flush_mode()
            return self.umeshes.update()

    def pick_unwrap(self, **unwrap_kwargs):
        hit = utypes.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            for isl in utypes.AdvIslands.calc_visible_with_mark_seam(umesh):
                hit.find_nearest_island(isl)

        if not hit or (self.max_distance < hit.min_dist):
            self.report({'WARNING'}, 'Island not found within a given radius')
            return {'CANCELLED'}

        isl = hit.island
        if utils.USE_GENERIC_UV_SYNC and isl.umesh.sync:
            if isl.umesh.elem_mode in ('VERT', 'EDGE'):
                isl.umesh.sync_from_mesh_if_needed()

        isl.select = True

        shared_selected_faces = []
        pinned_crn_uvs = []
        if not utils.USE_GENERIC_UV_SYNC and isl.umesh.sync:
            if isl.umesh.elem_mode in ('VERT', 'EDGE'):
                if isl.umesh.total_face_sel != len(isl):
                    faces = set(isl)
                    uv = isl.umesh.uv
                    for f in isl.umesh.bm.faces:
                        if f.select and f not in faces:
                            shared_selected_faces.append(f)
                            for crn in f.loops:
                                crn_uv = crn[uv]
                                if not crn_uv.pin_uv:
                                    crn_uv.pin_uv = True
                                    pinned_crn_uvs.append(crn_uv)

        unique_number_for_multiply = hash(isl[0])  # multiplayer
        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

        isl.umesh.value = isl.umesh.check_uniform_scale(report=self.report)
        isl.umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
        isl.apply_aspect_ratio()
        save_t = isl.save_transform()
        save_t.save_coords(self.unwrap_along, self.blend_factor)

        if self.mark_seam_inner_island:
            isl.mark_seam(additional=True)
        else:
            islands = utypes.AdvIslands([isl], isl.umesh)
            islands.indexing()
            isl.mark_seam_by_index(additional=True)

        bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)

        save_t.inplace(self.unwrap_along)
        save_t.apply_saved_coords(self.unwrap_along, self.blend_factor)
        is_updated = isl.reset_aspect_ratio()

        isl.select = False

        if utils.USE_GENERIC_UV_SYNC and isl.umesh.sync:
            if isl.umesh.elem_mode in ('VERT', 'EDGE'):
                isl.umesh.bm.uv_select_sync_valid = False

        if shared_selected_faces or pinned_crn_uvs or is_updated:
            for f in shared_selected_faces:
                f.select = False
            for crn_uv in pinned_crn_uvs:
                crn_uv.pin_uv = False

            isl.umesh.update()
        return {'FINISHED'}

    # TODO: Implement has unlinked_and_linked_selected_edges
    # TODO: Improve behavior self island unwrap
    @staticmethod
    def has_unlinked_and_linked_selected_faces(f_, uv, idx):
        unlinked_has_selected_face = False
        linked_has_selected_face = False
        for crn_ in f_.loops:
            first_co = crn_[uv].uv
            for l_crn in crn_.vert.link_loops:
                if l_crn.face.index == idx:
                    if l_crn[uv].uv == first_co:
                        if l_crn.face.select:
                            linked_has_selected_face = True
                else:
                    if l_crn.face.select:
                        unlinked_has_selected_face = True
        return unlinked_has_selected_face, linked_has_selected_face

    def unwrap_sync_verts_edges(self, **unwrap_kwargs):
        unique_number_for_multiply = 0
        pin_and_inplace = []
        unwrap_data: list[UnwrapData] = []
        for umesh in self.umeshes:
            uv = umesh.uv
            umesh.value = umesh.check_uniform_scale(report=self.report)
            umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
            # TODO: Full select unselected verts (with pins) of island for avoid incorrect behavior for relax OT
            islands = utypes.AdvIslands.calc_extended_any_elem_with_mark_seam(umesh)
            islands.indexing()

            for isl in islands:
                if unwrap_kwargs:
                    unique_number_for_multiply += hash(isl[0])  # multiplayer
                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)

            unpin_uvs = set()
            faces_to_select = set()
            verts_to_select = set()

            # Extend selected
            for idx, isl in enumerate(islands):
                for f in isl:
                    if f.select:
                        continue
                    # TODO: Implement skip only border face select, when inner selected vert face exist
                    # Skip full selected and full unselected
                    if sum(v.select for v in f.verts) not in (0, len(f.verts)):
                        unlinked_sel, linked_sel = self.has_unlinked_and_linked_selected_faces(f, uv, idx)
                        # If there is a linked select face or there are selected only verts, then unwrap it
                        if linked_sel or not (unlinked_sel or linked_sel):
                            faces_to_select.add(f)
                            for v in f.verts:
                                if not v.select:
                                    verts_to_select.add(v)
                        else:
                            # If only the unlinked face is selected, then pin it.
                            for crn_ in f.loops:
                                for l_crn_ in linked_crn_uv_by_island_index_unordered_included(crn_, uv, idx):
                                    crn_uv = l_crn_[uv]
                                    if not crn_uv.pin_uv:
                                        crn_uv.pin_uv = True
                                        unpin_uvs.add(crn_uv)

            for f in faces_to_select:
                f.select = True

            for v in verts_to_select:
                v.select = True
                for crn in v.link_loops:
                    crn_uv = crn[uv]
                    if not crn_uv.pin_uv:
                        crn_uv.pin_uv = True
                        unpin_uvs.add(crn_uv)

            if self.umeshes.elem_mode == 'EDGE':  # EDGE
                for e in umesh.bm.edges:
                    e.select = sum(v.select for v in e.verts) == 2

            save_transform_islands = []
            for isl in islands:
                if any(v.select for f in isl for v in f.verts):
                    isl.apply_aspect_ratio()
                    save_t = isl.save_transform()
                    save_t.save_coords(self.unwrap_along, self.blend_factor)
                    save_transform_islands.append(save_t)

            pin_and_inplace.append((unpin_uvs, save_transform_islands))
            unwrap_data.append(UnwrapData(umesh, unpin_uvs, save_transform_islands, verts_to_select))

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)
        bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)

        for ud in unwrap_data:
            for pin in ud.pins:
                pin.pin_uv = False
            for isl in ud.islands:
                isl.inplace(self.unwrap_along)
                isl.apply_saved_coords(self.unwrap_along, self.blend_factor)
                isl.island.reset_aspect_ratio()
            for v in ud.temp_selected:
                v.select = False

            if self.umeshes.elem_mode == 'EDGE':  # EDGE
                for e in ud.umesh.bm.edges:
                    e.select = sum(v.select for v in e.verts) == 2

            if self.unwrap == 'MINIMUM_STRETCH':
                if self.umeshes.elem_mode != 'FACE':
                    # It might be worth bug reporting this moment when SLIM causes a "grow effect"
                    ud.umesh.bm.select_flush(False)

    @staticmethod
    def unwrap_sync_faces_extend_select_and_set_pins(isl):
        to_select = []
        unpinned = []
        uv = isl.umesh.uv
        sync = isl.umesh.sync
        for f in isl:
            if f.select:
                continue

            has_selected_linked_faces = False
            temp_static = []
            for crn in f.loops:
                linked = utils.linked_crn_to_vert_pair_with_seam(crn, uv, sync)
                if any(cc.face.select for cc in linked):
                    has_selected_linked_faces = True
                else:
                    temp_static.append(crn)

            if has_selected_linked_faces:
                to_select.append(f)

                for crn in temp_static:
                    crn_uv = crn[uv]
                    if not crn_uv.pin_uv:
                        crn_uv.pin_uv = True
                    unpinned.append(crn_uv)
        for f in to_select:
            f.select = True
        isl.sequence = (unpinned, to_select)

    def unwrap_sync_faces(self, **unwrap_kwargs):
        assert self.umeshes.elem_mode == 'FACE'
        unique_number_for_multiply = 0

        all_transform_islands = []
        for umesh in reversed(self.umeshes):
            umesh.value = umesh.check_uniform_scale(report=self.report)
            umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
            islands_extended = utypes.AdvIslands.calc_extended_with_mark_seam(umesh)
            islands_extended.indexing()

            for isl in islands_extended:
                if unwrap_kwargs:
                    unique_number_for_multiply += hash(isl[0])  # multiplayer

                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)

                self.unwrap_sync_faces_extend_select_and_set_pins(isl)

                isl.apply_aspect_ratio()
                save_t = isl.save_transform()
                save_t.save_coords(self.unwrap_along, self.blend_factor)
                all_transform_islands.append(save_t)

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)
        bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)

        for isl in all_transform_islands:
            unpinned, to_select = isl.island.sequence
            for pin in unpinned:
                pin.pin_uv = False
            for f in to_select:
                f.select = False

            isl.inplace(self.unwrap_along)
            isl.apply_saved_coords(self.unwrap_along, self.blend_factor)
            isl.island.reset_aspect_ratio()

    def unwrap_non_sync(self, **unwrap_kwargs):
        save_transform_islands = []
        unique_number_for_multiply = 0

        tool_settings = bpy.context.scene.tool_settings
        is_sticky_mode_disabled = tool_settings.uv_sticky_select_mode == 'DISABLED'

        for umesh in reversed(self.umeshes):
            uv = umesh.uv
            umesh.value = umesh.check_uniform_scale(report=self.report)
            umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
            islands = utypes.AdvIslands.calc_extended_any_elem_with_mark_seam(umesh)
            if not self.mark_seam_inner_island:
                islands.indexing()

            for isl in islands:
                if unwrap_kwargs:
                    unique_number_for_multiply += hash(isl[0])  # multiplayer

                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)

            if is_sticky_mode_disabled:
                face_select_get = utils.face_select_get_func(umesh)
                crn_select_get = utils.vert_select_get_func(umesh)
                for isl in islands:
                    unpin_uvs = set()
                    corners_to_select = set()
                    for f in isl:
                        if face_select_get(f):
                            continue

                        temp_static = []
                        has_selected = False
                        for crn in f.loops:
                            if crn_select_get(crn):
                                continue
                            linked = utils.linked_crn_to_vert_pair_with_seam(crn, umesh.uv, umesh.sync)
                            if any(crn_select_get(c) for c in linked):
                                has_selected = True
                                corners_to_select.add(crn[uv])
                            else:
                                temp_static.append(crn)
                        if has_selected:
                            for cc in temp_static:
                                cc_uv = cc[uv]
                                if not cc_uv.pin_uv:
                                    unpin_uvs.add(cc_uv)
                                    corners_to_select.add(cc_uv)

                    for unpin_crn in unpin_uvs:
                        unpin_crn.pin_uv = True
                    for unsel_crn in corners_to_select:
                        unsel_crn.select = True
                    isl.sequence = (unpin_uvs, corners_to_select)

            for isl in islands:
                isl.apply_aspect_ratio()
                save_t = isl.save_transform()
                save_t.save_coords(self.unwrap_along, self.blend_factor)
                save_transform_islands.append(save_t)

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

        bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)

        for isl in save_transform_islands:
            isl.inplace(self.unwrap_along)
            isl.apply_saved_coords(self.unwrap_along, self.blend_factor)
            isl.island.reset_aspect_ratio()

            if is_sticky_mode_disabled:
                if isl.island.sequence:
                    unpin_uvs, corners_to_select = isl.island.sequence
                    for unpin_crn in unpin_uvs:
                        unpin_crn.pin_uv = False
                    for unsel_crn in corners_to_select:
                        unsel_crn.select = False

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


class UNIV_OT_Unwrap_VIEW3D(bpy.types.Operator, utypes.RayCast):
    bl_idname = "mesh.univ_unwrap"
    bl_label = "Unwrap"
    bl_description = ("Inplace unwrap the mesh of object being edited\n\n "
                      "Organic Mode has incorrect behavior with pinned and flipped islands")
    bl_options = {'REGISTER', 'UNDO'}

    unwrap: bpy.props.EnumProperty(name='Unwrap',
                                   default='ANGLE_BASED',
                                   items=(('ANGLE_BASED', 'Hard Surface', ''),
                                          ('CONFORMAL', 'Conformal', ''),
                                          ('MINIMUM_STRETCH', 'Organic', '')))

    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(self, 'use_correct_aspect')
        self.layout.row(align=True).prop(self, 'unwrap', expand=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            self.init_data_for_ray_cast(event)
            return self.execute(context)
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utypes.RayCast.__init__(self)
        self.umeshes: utypes.UMeshes | None = None
        self.texel = -1
        self.texture_size = -1

    def execute(self, context):
        self.umeshes = utypes.UMeshes.calc(self.report, verify_uv=False)

        self.umeshes.fix_context()
        self.umeshes.set_sync()
        self.umeshes.sync_invalidate()

        from ..preferences import univ_settings
        self.texel = univ_settings().texel_density
        self.texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2

        if self.use_correct_aspect:
            self.umeshes.calc_aspect_ratio(from_mesh=True)

        if self.unwrap == 'MINIMUM_STRETCH' and bpy.app.version < (4, 3, 0):
            self.unwrap = 'ANGLE_BASED'
            self.report({'WARNING'}, 'Organic Mode is not supported in Blender versions below 4.3')

        selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_verts()
        self.umeshes = selected_umeshes if selected_umeshes else unselected_umeshes
        if not self.umeshes:
            return self.umeshes.update()

        if not selected_umeshes and self.mouse_pos_from_3d:
            return self.pick_unwrap()
        else:
            for u in reversed(self.umeshes):
                if not u.has_uv and not u.total_face_sel:
                    self.umeshes.umeshes.remove(u)
            if not self.umeshes:
                self.report({'WARNING'}, 'Need selected faces for objects without uv')
                return {'CANCELLED'}

            self.unwrap_selected()
            self.umeshes.update()
            return {'FINISHED'}

    def pick_unwrap(self, **unwrap_kwargs):
        if not (hit := self.ray_cast(prefs().max_pick_distance)):
            return {'CANCELLED'}

        umesh = hit.umesh
        umesh.value = umesh.check_uniform_scale(report=self.report)
        if umesh.has_uv:
            umesh.verify_uv()
            mesh_island, mesh_isl_set = hit.calc_mesh_island_with_seam()
            adv_subislands = mesh_island.calc_adv_subislands_with_mark_seam()
            for isl in adv_subislands:
                isl.select = True

            shared_selected_faces = []
            pinned_crn_uvs = []
            # In vert/edge selection mode, you can accidentally select extra faces.
            # To avoid this, we pin them.
            if umesh.total_face_sel != len(mesh_isl_set):
                uv = umesh.uv
                for f in umesh.bm.faces:
                    if f.select and f not in mesh_isl_set:
                        shared_selected_faces.append(f)
                        for crn in f.loops:
                            crn_uv = crn[uv]
                            if not crn_uv.pin_uv:
                                crn_uv.pin_uv = True
                                pinned_crn_uvs.append(crn_uv)

            unique_number_for_multiply = hash(mesh_island[0])  # multiplayer
            UNIV_OT_Unwrap.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

            for isl in adv_subislands:
                isl.apply_aspect_ratio()
            save_t = utypes.SaveTransform(adv_subislands)

            bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)
            umesh.verify_uv()

            adv_island = mesh_island.to_adv_island()
            save_t.island = adv_island
            save_t.inplace()

            adv_island.reset_aspect_ratio()
            adv_island.select = False

            for f in shared_selected_faces:
                f.select = False
            for crn_uv in pinned_crn_uvs:
                crn_uv.pin_uv = False
        else:
            mesh_island, mesh_isl_set = hit.calc_mesh_island_with_seam()
            adv_island = mesh_island.to_adv_island()
            adv_island.select = True

            bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)

            umesh.verify_uv()
            unique_number_for_multiply = hash(mesh_island[0])  # multiplayer
            UNIV_OT_Unwrap.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

            adv_island.calc_area_uv()
            adv_island.calc_area_3d(scale=umesh.value)

            if (status := adv_island.set_texel(self.texel, self.texture_size)) is None:  # noqa
                # zero_area_islands.append(isl)
                pass

            # reset aspect
            scale = Vector((1 / umesh.aspect, 1))
            adv_island.scale(scale, adv_island.bbox.center)
            adv_island.select = False

        umesh.update()
        return {'FINISHED'}

    def unwrap_selected(self, **unwrap_kwargs):
        meshes_with_uvs = []
        meshes_without_uvs = []
        unique_number = 0
        for umesh in self.umeshes:
            umesh.value = umesh.check_uniform_scale(report=self.report)
            if not umesh.has_uv:
                meshes_without_uvs.append(umesh)
            else:
                umesh.verify_uv()
                meshes_with_uvs.append(umesh)
                if self.umeshes.elem_mode == 'VERT':
                    if umesh.total_face_sel:
                        unique_number += self.unwrap_selected_faces_preprocess_vert_edge_mode(umesh)
                    else:
                        unique_number += self.unwrap_selected_verts(umesh)
                elif self.umeshes.elem_mode == 'EDGE':
                    if umesh.total_face_sel:
                        unique_number += self.unwrap_selected_faces_preprocess_vert_edge_mode(umesh)
                    else:
                        unique_number += self.unwrap_selected_edges(umesh)
                else:
                    unique_number += self.unwrap_selected_faces_preprocess(umesh)

        UNIV_OT_Unwrap.multiply_relax(unique_number % (1 << 62), unwrap_kwargs)
        bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)

        for umesh in meshes_with_uvs:
            self.unwrap_selected_faces_postprocess(umesh)
            umesh.bm.select_flush(False)

        self.unwrap_without_uvs(meshes_without_uvs)

    def unwrap_selected_faces_preprocess_vert_edge_mode(self, umesh):
        assert umesh.total_face_sel
        assert self.umeshes.elem_mode in ('VERT', 'EDGE')
        mesh_islands = utypes.MeshIslands.calc_visible_with_mark_seam(umesh)
        unique_number = 0
        pinned = []
        to_select = []
        without_selection_islands = []
        save_transform_islands = []

        uv = umesh.uv
        for mesh_isl in mesh_islands:
            if not any(f.select for f in mesh_isl):
                without_selection_islands.append(mesh_isl)
                continue
            unique_number += hash(mesh_isl[0])
            adv_islands = mesh_isl.calc_adv_subislands_with_mark_seam()
            adv_islands.apply_aspect_ratio()
            safe_transform = utypes.SaveTransform(adv_islands)
            save_transform_islands.append(safe_transform)

            for f in mesh_isl:
                if f.select:
                    continue
                to_select.append(f)
                for crn in f.loops:
                    crn_uv = crn[uv]
                    if crn_uv.pin_uv:
                        continue

                    if crn.vert.select:
                        # If linked faces are selected, then crn should unwrap as well
                        if any(crn_.face.select for crn_ in utils.linked_crn_to_vert_with_seam_3d_iter(crn)):
                            continue
                    crn_uv.pin_uv = True
                    pinned.append(crn_uv)

        expected_total_selected_faces = umesh.total_face_sel + len(to_select)
        if self.umeshes.elem_mode == 'VERT':
            to_deselect_elements = [v for f in to_select for v in f.verts if not v.select]
        else:
            to_deselect_elements = [e for f in to_select for e in f.edges if not e.select]

        for f in to_select:
            f.select = True

        # May select faces from other islands, if so pin them and safe face to unselect
        if expected_total_selected_faces != umesh.total_face_sel:
            for isl in without_selection_islands:
                for f in isl:
                    if f.select:
                        to_deselect_elements.append(f)
                        for crn in f.loops:
                            crn_uv = crn[uv]
                            if not crn_uv.pin_uv:
                                pinned.append(crn_uv)

        umesh.other = UnwrapData(None, pinned, save_transform_islands, to_deselect_elements)
        return unique_number

    def unwrap_selected_verts(self, umesh):
        assert not umesh.total_face_sel
        assert self.umeshes.elem_mode == 'VERT'
        mesh_islands = utypes.MeshIslands.calc_visible_with_mark_seam(umesh)
        unique_number = 0
        pinned = []
        to_select = []
        without_selection_islands = []
        save_transform_islands = []

        uv = umesh.uv
        for mesh_isl in mesh_islands:
            if not any(v.select for f in mesh_isl for v in f.verts):
                without_selection_islands.append(mesh_isl)
                continue

            unique_number += hash(mesh_isl[0])
            adv_islands = mesh_isl.calc_adv_subislands_with_mark_seam()
            adv_islands.apply_aspect_ratio()
            safe_transform = utypes.SaveTransform(adv_islands)
            save_transform_islands.append(safe_transform)
            to_select.extend(mesh_isl)

            for f in mesh_isl:
                for crn in f.loops:
                    if crn.vert.select:
                        continue

                    crn_uv = crn[uv]
                    if crn_uv.pin_uv:
                        continue

                    crn_uv.pin_uv = True
                    pinned.append(crn_uv)

        expected_total_selected_faces = umesh.total_face_sel + len(to_select)
        to_deselect_elements = [v for f in to_select for v in f.verts if not v.select]

        for f in to_select:
            f.select = True

        # May select faces from other islands, if so pin them and safe face to unselect
        if expected_total_selected_faces != umesh.total_face_sel:
            for isl in without_selection_islands:
                for f in isl:
                    if f.select:
                        to_deselect_elements.append(f)
                        for crn in f.loops:
                            crn_uv = crn[uv]
                            if not crn_uv.pin_uv:
                                pinned.append(crn_uv)

        umesh.other = UnwrapData(None, pinned, save_transform_islands, to_deselect_elements)
        return unique_number

    def unwrap_selected_edges(self, umesh):
        assert not umesh.total_face_sel
        assert self.umeshes.elem_mode == 'EDGE'
        mesh_islands = utypes.MeshIslands.calc_visible_with_mark_seam(umesh)
        unique_number = 0
        pinned = []
        to_select = []
        without_selection_islands = []
        save_transform_islands = []

        uv = umesh.uv
        for mesh_isl in mesh_islands:
            if not any(e.select for f in mesh_isl for e in f.edges):
                without_selection_islands.append(mesh_isl)
                continue
            unique_number += hash(mesh_isl[0])
            adv_islands = mesh_isl.calc_adv_subislands_with_mark_seam()
            adv_islands.apply_aspect_ratio()
            safe_transform = utypes.SaveTransform(adv_islands)
            save_transform_islands.append(safe_transform)
            to_select.extend(mesh_isl)

            for f in mesh_isl:
                if all(not v.select for v in f.verts):
                    for crn in f.loops:
                        crn_uv = crn[uv]
                        if crn_uv.pin_uv:
                            continue
                        crn_uv.pin_uv = True
                        pinned.append(crn_uv)
                    continue

                for crn in f.loops:
                    if crn.edge.select:
                        continue
                    crn_uv = crn[uv]
                    if crn.vert.select:
                        if crn_uv.pin_uv:
                            continue
                        if any(crn_.edge.select for crn_ in utils.linked_crn_to_vert_with_seam_3d_iter(crn)):
                            continue

                    if crn_uv.pin_uv:
                        continue
                    crn_uv.pin_uv = True
                    pinned.append(crn_uv)

        expected_total_selected_faces = umesh.total_face_sel + len(to_select)
        to_deselect_elements = [e for f in to_select for e in f.edges if not e.select]

        for f in to_select:
            f.select = True

        # May select faces from other islands, if so pin them and safe face to unselect
        if expected_total_selected_faces != umesh.total_face_sel:
            for isl in without_selection_islands:
                for f in isl:
                    if f.select:
                        to_deselect_elements.append(f)
                        for crn in f.loops:
                            crn_uv = crn[uv]
                            if not crn_uv.pin_uv:
                                pinned.append(crn_uv)

        umesh.other = UnwrapData(None, pinned, save_transform_islands, to_deselect_elements)
        return unique_number

    @staticmethod
    def unwrap_selected_faces_preprocess(umesh):
        assert umesh.total_face_sel
        mesh_islands = utypes.MeshIslands.calc_extended_with_mark_seam(umesh)
        unique_number = 0
        pinned = []
        to_select = []
        save_transform_islands = []

        uv = umesh.uv
        for mesh_isl in mesh_islands:
            unique_number += hash(mesh_isl[0])
            adv_islands = mesh_isl.calc_adv_subislands_with_mark_seam()
            adv_islands.apply_aspect_ratio()
            safe_transform = utypes.SaveTransform(adv_islands)
            save_transform_islands.append(safe_transform)

            for f in mesh_isl:
                if f.select:
                    continue
                to_select.append(f)
                for crn in f.loops:
                    crn_uv = crn[uv]
                    if crn_uv.pin_uv:
                        continue

                    if crn.vert.select:
                        # If linked faces are selected, then crn should unwrap as well
                        if any(crn_.face.select for crn_ in utils.linked_crn_to_vert_with_seam_3d_iter(crn)):
                            continue
                    crn_uv.pin_uv = True
                    pinned.append(crn_uv)

        for f in to_select:
            f.select = True
        umesh.other = UnwrapData(None, pinned, save_transform_islands, to_select)
        return unique_number

    @staticmethod
    def unwrap_selected_faces_postprocess(umesh):
        unwrap_data: UnwrapData = umesh.other
        for f in unwrap_data.temp_selected:
            f.select = False
        for crn_uv in unwrap_data.pins:
            crn_uv.pin_uv = False

        for safe_transform in unwrap_data.islands:
            safe_transform.inplace_mesh_island()
            safe_transform.island.reset_aspect_ratio()

    def unwrap_without_uvs(self, umeshes):
        for umesh in umeshes:
            mesh_islands = utypes.MeshIslands.calc_extended_with_mark_seam(umesh)
            umesh.verify_uv()

            adv_islands = mesh_islands.to_adv_islands()

            adv_islands.calc_area_uv()
            adv_islands.calc_area_3d(scale=umesh.value)

            # reset aspect
            for adv_isl in adv_islands:
                adv_isl.set_texel(self.texel, self.texture_size)
                scale = Vector((1 / umesh.aspect, 1))
                adv_isl.scale(scale, adv_isl.bbox.center)
