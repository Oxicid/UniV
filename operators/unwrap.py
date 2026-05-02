# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
from mathutils import Vector
from bmesh.types import BMLoop

from .. import utypes
from .. import utils
from ..preferences import prefs, univ_settings
from ..utils import linked_crn_uv_by_island_index_unordered_included


class UnwrapData:
    def __init__(self, umesh, pins, island, selected):
        self.umesh: utypes.UMesh = umesh
        self.pins = pins
        self.islands: list[utypes.SaveTransform] = island
        self.temp_selected = selected


MULTIPLAYER = 1
UNIQUE_NUMBER_FOR_MULTIPLY = -1


# noinspection PyTypeHints
class UNIV_OT_Unwrap(bpy.types.Operator):
    bl_idname = "uv.univ_unwrap"
    bl_label = "Unwrap"
    bl_description = ("Inplace unwrap the mesh of object being edited\n\n "
                      "Organic Mode has incorrect behavior with pinned and flipped islands")
    bl_options = {'REGISTER', 'UNDO'}

    unwrap: bpy.props.EnumProperty(name='Unwrap', default='ANGLE_BASED',
                                   items=(('ANGLE_BASED', 'Hard Surface', ''),
                                          ('CONFORMAL', 'Conformal', ''),
                                          ('MINIMUM_STRETCH', 'Organic', '')))
    blend_factor: bpy.props.FloatProperty(name='Blend Factor', default=1, soft_min=0, soft_max=1)
    fill_holes: bpy.props.BoolProperty(name='Fill Holes', default=True)
    mark_seam_inner_island: bpy.props.BoolProperty(name='Mark Seam Self Borders', default=True,
                                    description='Marking seams where there are split edges within the same island.')
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)
    constraints_weight: bpy.props.FloatProperty(name='Constraints Weight', default=0, min=0, max=0, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        self.layout.prop(univ_settings(), 'use_texel')
        self.layout.prop(self, 'fill_holes')

        self.layout.prop(self, 'use_correct_aspect')
        self.layout.prop(self, 'mark_seam_inner_island')
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

        selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_by_context()
        self.umeshes = selected_umeshes if selected_umeshes else unselected_umeshes
        if not self.umeshes:
            return self.umeshes.update()

        if not selected_umeshes and self.max_distance is not None and context.area.ui_type == 'UV':
            return self.pick_unwrap()
        else:
            if not selected_umeshes:
                self.report({'WARNING'}, 'Need selected geometry')
                return {'CANCELLED'}

            if self.umeshes.sync:
                if self.umeshes.elem_mode == 'FACE':
                    self.unwrap_sync_faces()
                else:
                    self.unwrap_sync_verts_or_edges()
            else:
                self.unwrap_non_sync()

            # for umesh in self.umeshes:
            #     if not umesh.sync_valid:
            #         umesh.bm.select_flush_mode()
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
        unique_number_for_multiply = hash(isl[0])  # multiplayer
        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

        isl.umesh.value = isl.umesh.check_uniform_scale(report=self.report)
        isl.umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0


        # Constraints system
        ##################################################
        from importlib.util import find_spec
        found_univ_pro = find_spec(f"{__package__.rpartition('.')[0]}.univ_pro") is not None

        if found_univ_pro and self.bl_label == 'Unwrap' and self.constraints_weight and isl.has_constraints_edge():
            self.pick_unwrap_by_constraints(isl)
            return {'FINISHED'}
        ##################################################

        if utils.USE_GENERIC_UV_SYNC and isl.umesh.sync:
            if isl.umesh.elem_mode in ('VERT', 'EDGE'):
                isl.umesh.sync_from_mesh_if_needed()

        isl.select = True

        accidentally_selected_faces = []
        if not utils.USE_GENERIC_UV_SYNC and isl.umesh.sync:
            accidentally_selected_faces = self.prepare_accidentally_selected_islands_for_pick(isl)
            for f in accidentally_selected_faces:
                f.hide = True

        isl.apply_aspect_ratio()
        save_t = isl.save_transform(flip_if_needed=True)
        save_t.save_coords(self.blend_factor)

        if self.mark_seam_inner_island:
            isl.mark_seam(additional=True)
        else:
            islands = utypes.AdvIslands([isl], isl.umesh)
            islands.indexing()
            isl.mark_seam_by_index(additional=True)

        bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)

        save_t.inplace(flip_if_needed=False)
        save_t.apply_saved_coords(self.blend_factor, flip_if_needed=True)

        isl.reset_aspect_ratio()

        if save_t.rotate:
            utils.set_global_texel(save_t.island)

        isl.select = False

        if utils.USE_GENERIC_UV_SYNC and isl.umesh.sync:
            if isl.umesh.elem_mode in ('VERT', 'EDGE'):
                isl.umesh.bm.uv_select_sync_valid = False

        for f in accidentally_selected_faces:
            f.hide = False
            f.select = False

        isl.umesh.update()
        return {'FINISHED'}

    def pick_unwrap_by_constraints(self, isl):
        raise

    @staticmethod
    def prepare_accidentally_selected_islands_for_pick(isl):
        """
        In the old sync system, extra faces could be accidentally selected.
        They are pinned to prevent any effect, and the elements are saved so their flags can be restored.
        """
        if isl.umesh.elem_mode in ('VERT', 'EDGE'):
            if isl.umesh.total_face_sel != len(isl):  # Fast check if island single.
                faces_set = set(isl)
                return [f for f in isl.umesh.bm.faces if f.select and f not in faces_set]
        return []

    @staticmethod
    def is_accidentally_selected_crn(crn: BMLoop):
        assert not crn.face.select
        assert crn.vert.select
        linked_corners = utils.linked_crn_to_vert_without_coord_check_with_seam_for_sync_unwrap(crn)
        # Has linked selected face.
        if any(l_crn.face.select for l_crn in linked_corners):
            return False

        all_linked = crn.vert.link_loops

        # Is inner vertex
        if len(linked_corners) == len(all_linked):
            return False

        # Has unlinked selected face.
        for ll_crn in all_linked:
            if ll_crn.face.select:
                return True
        return False


    def unwrap_sync_verts_or_edges(self, **unwrap_kwargs):
        from importlib.util import find_spec
        found_univ_pro = find_spec(f"{__package__.rpartition('.')[0]}.univ_pro") is not None

        has_native_unwrapped = 0
        failed_total = 0
        unique_number_for_multiply = 0
        to_lock_constraints_islands = []

        unwrap_data: list[UnwrapData] = []
        full_selected_meshes = []
        all_pins = []

        # TODO: Add update tag
        for umesh in self.umeshes:
            has_full_selected_uv_faces = umesh.has_full_selected_uv_faces()
            if has_full_selected_uv_faces:
                full_selected_meshes.append(umesh)

            uv = umesh.uv
            sync = umesh.sync

            crn_select_get = utils.vert_select_get_func(umesh)
            face_select_get = utils.face_select_get_func(umesh)

            umesh.value = umesh.check_uniform_scale(report=self.report)
            umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
            # TODO: Full select unselected verts (with pins) of island for avoid incorrect behavior for relax OT
            islands = utypes.AdvIslands.calc_extended_any_elem_with_mark_seam(umesh)
            islands.indexing()

            for isl in islands:
                unique_number_for_multiply += hash(isl[0])  # multiplayer
                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)

            unpin_uvs = set()
            faces_to_select = set()
            verts_to_lock = set()

            # Extend selection

            for idx, isl in enumerate(islands):
                isl.tag = False  # tagged = native unwrap

                # Constraints system
                ##################################################
                if found_univ_pro and self.bl_label == 'Unwrap' and self.constraints_weight and isl.has_constraints_edge(selected=True):
                    if utils.USE_GENERIC_UV_SYNC and umesh.sync_valid:
                        for f in isl:
                            if face_select_get(f):
                                for crn in f.loops:
                                    crn.tag = not crn[uv].pin_uv
                            else:
                                for crn in f.loops:
                                    if crn[uv].pin_uv:
                                        crn.tag = False
                                    else:
                                        linked = utils.linked_crn_to_vert_pair_with_seam(crn, uv, sync)
                                        crn.tag = any(crn_select_get(cc) for cc in linked)
                    elif has_full_selected_uv_faces:
                        for crn in isl.corners_iter():
                            crn.tag = not crn[uv].pin_uv
                    else:  # Invalid sync case.
                        for f in isl:
                            if f.select:
                                for crn in f.loops:
                                    crn.tag = not crn[uv].pin_uv
                            else:
                                # TODO: Check accidentally selected for edges (and fix some case in 'has_constraints_edge')
                                for crn in f.loops:
                                    if crn[uv].pin_uv or not crn.vert.select:
                                        crn.tag = False
                                    else:
                                        crn.tag = not self.is_accidentally_selected_crn(crn)

                    is_static = self.is_static_island(isl)

                    with utils.uv_parametrizer.unwrap_time_report(self.report):
                        failed_total += utils.uv_parametrizer.unwrap_isl_by_tag(isl,
                                                                            unwrap_along=self.unwrap_along,  # noqa
                                                                            use_abf=self.unwrap == 'ANGLE_BASED',
                                                                            topology_from_uvs=self.mark_seam_inner_island,
                                                                            blend_factor=self.blend_factor,
                                                                            fill_holes=self.fill_holes,
                                                                            constraints_factor=self.constraints_weight * 100)
                    to_lock_constraints_islands.append(isl)
                    isl.reset_aspect_ratio()

                    if not is_static:
                        utils.set_global_texel(isl)

                ##################################################
                else:
                    isl.tag = True
                    if has_full_selected_uv_faces:
                        continue

                    if umesh.sync_valid:
                        for f in isl:
                            # Skip full selected and full unselected
                            if f.select or not any(crn.uv_select_vert for crn in f.loops):
                                continue
                            verts_to_lock.update(crn for crn in f.loops if not crn.uv_select_vert)
                    else:
                        # TODO: Add this to 'has_constraints_edge'
                        if umesh.elem_mode == 'VERT':
                            for f in isl:
                                # Skip full selected and full unselected
                                if f.select:
                                    continue
                                for crn in f.loops:
                                    if not crn.vert.select:
                                        continue

                                    if self.is_accidentally_selected_crn(crn):
                                        # If only the unlinked face is selected, then pin it.
                                        for l_crn in linked_crn_uv_by_island_index_unordered_included(crn, uv, idx):
                                            crn_uv = l_crn[uv]
                                            if not crn_uv.pin_uv:
                                                crn_uv.pin_uv = True
                                                unpin_uvs.add(crn_uv)
                                    else:
                                        # Here it is important to delay selecting the face so that the other vertices
                                        # can correctly compute pins and accidentally selected vertices.
                                        faces_to_select.add(f)
                                        verts_to_lock.update(v for v in f.verts if not v.select)
                        else:
                            def has_special_selected_edges(linked_crn):
                                # When there is a linked selected face, do not pin this crn.
                                if linked_crn.face.select:
                                    return True
                                # If there is no seam or edge boundary or face selected for the pair crn,
                                # then it means that this is not an accidentally selected edge.
                                if linked_crn.edge.select:
                                    if not linked_crn.edge.seam or linked_crn.edge.is_boundary:
                                        return True
                                    if not linked_crn.link_loop_radial_prev.face.select:
                                        return True

                                prev = linked_crn.link_loop_prev
                                if prev.edge.select:
                                    if not prev.edge.seam or prev.edge.is_boundary:
                                        return True
                                    if not prev.link_loop_radial_prev.face.select:
                                        return True
                                return False

                            linked_to_vert = utils.linked_crn_to_vert_without_coord_check_with_seam_for_sync_unwrap

                            for f in isl:
                                # Skip full selected and full unselected
                                if f.select or all(not crn.vert.select for crn in f.loops):
                                    continue

                                static_corners = set()
                                unwrap_corners = set()
                                for crn in f.loops:
                                    if not crn.vert.select or crn in unwrap_corners or crn in static_corners:
                                        continue

                                    # Single vertex select case
                                    if not crn.edge.select:
                                        if not crn.link_loop_prev.edge.select:
                                            if any(has_special_selected_edges(l_crn) for l_crn in linked_to_vert(crn)):
                                                unwrap_corners.add(crn)
                                            else:
                                                static_corners.add(crn)

                                    else:  # Edge select case
                                        if any(has_special_selected_edges(l_crn) for l_crn in [crn]+linked_to_vert(crn)):
                                            unwrap_corners.add(crn)
                                        else:
                                            static_corners.add(crn)

                                        next_crn = crn.link_loop_next
                                        if next_crn in unwrap_corners:
                                            continue
                                        if next_crn in static_corners:
                                            continue

                                        if any(has_special_selected_edges(l_crn) for l_crn in [next_crn]+linked_to_vert(next_crn)):
                                            unwrap_corners.add(crn)
                                        else:
                                            static_corners.add(crn)

                                for cc in static_corners:
                                    crn_uv = cc[uv]
                                    if not crn_uv.pin_uv:
                                        crn_uv.pin_uv = True
                                        unpin_uvs.add(crn_uv)

                                if unwrap_corners:
                                    faces_to_select.add(f)
                                    verts_to_lock.update(v for v in f.verts if not v.select)


            to_restore_selection = set()
            if not has_full_selected_uv_faces:
                if umesh.sync_valid:
                    for crn in verts_to_lock:
                        if not crn.uv_select_vert:
                            crn.uv_select_vert = True
                            to_restore_selection.add(crn)
                        crn_uv = crn[uv]
                        if not crn_uv.pin_uv:
                            crn_uv.pin_uv = True
                            unpin_uvs.add(crn_uv)
                else:
                    if self.umeshes.elem_mode == 'EDGE':
                        to_restore_selection = {e for f in faces_to_select for e in f.edges if e.select}
                    else:
                        to_restore_selection = {v for f in faces_to_select for v in f.verts if v.select}

                    for f in faces_to_select:
                        f.select = True

                    # Extra vertices may accidentally be selected, so we pin them.
                    for v in verts_to_lock:
                        for crn in v.link_loops:
                            crn_uv = crn[uv]
                            if not crn_uv.pin_uv:
                                crn_uv.pin_uv = True
                                unpin_uvs.add(crn_uv)



            save_transform_islands = []
            for isl in islands:
                if isl.tag:
                    has_non_static_face = any(not all(c[uv].pin_uv for c in f.loops) for f in isl if f.select)
                    if has_non_static_face:
                        isl.apply_aspect_ratio()
                        save_t = isl.save_transform(flip_if_needed=True)
                        save_t.save_coords(self.blend_factor)
                        save_transform_islands.append(save_t)

            if save_transform_islands:
                has_native_unwrapped = True

            all_pins.append(unpin_uvs)
            unwrap_data.append(UnwrapData(umesh, faces_to_select, save_transform_islands, to_restore_selection))

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

        if has_native_unwrapped:
            for to_lock_isl in to_lock_constraints_islands:
                to_lock_isl.sequence = self.lock_island_from_unwrap_and_get_pins_sync(to_lock_isl)

            bpy.ops.uv.unwrap(method=self.unwrap, fill_holes=self.fill_holes, correct_aspect=False, **unwrap_kwargs)

            for to_lock_isl in to_lock_constraints_islands:
                for crn_uv in to_lock_isl.sequence:
                    crn_uv.pin_uv = False

            for ud in unwrap_data:
                for isl in ud.islands:
                    isl.inplace(flip_if_needed=False)
                    isl.apply_saved_coords(self.blend_factor, flip_if_needed=True)
                    isl.island.reset_aspect_ratio()

                    if isl.rotate:
                        utils.set_global_texel(isl.island)

                if ud.umesh not in full_selected_meshes:  # Skip full selected
                    if ud.umesh.sync_valid:
                        for crn in ud.temp_selected:
                            crn.uv_select_vert = False
                    else:
                        pass
                        selected_faces = ud.pins
                        for f in selected_faces:
                            f.select = False

                        to_restore = ud.temp_selected
                        for v_or_e in to_restore:
                            v_or_e.select = True

        for isl in to_lock_constraints_islands:
            isl.reset_aspect_ratio()

        for pins in all_pins:
            for pin in pins:
                pin.pin_uv = False

        if failed_total:
            self.report({'WARNING'}, f"It is not possible to unwrap {failed_total!r} islands. "
                                     f"Try again by setting at least one pin or by partially selecting the island.")


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
        if isl.umesh.sync_valid:
            true_select = []
            for f in to_select:
                # f.select = True
                # f.uv_select = True
                for crn in f.loops:
                    if not crn.uv_select_vert:
                        crn.uv_select_vert = True
                        true_select.append(crn)
            to_select = true_select
        else:
            for f in to_select:
                f.select = True
        isl.sequence = (unpinned, to_select)


    def unwrap_sync_faces(self, **unwrap_kwargs):
        assert self.umeshes.elem_mode == 'FACE'
        from importlib.util import find_spec
        found_univ_pro = find_spec(f"{__package__.rpartition('.')[0]}.univ_pro") is not None

        failed_total = 0
        unique_number_for_multiply = 0
        hidden_constraints_islands: list[utypes.AdvIsland] = []
        all_transform_islands = []
        for umesh in reversed(self.umeshes):
            umesh.value = umesh.check_uniform_scale(report=self.report)
            umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
            islands_extended = utypes.AdvIslands.calc_extended_with_mark_seam(umesh)
            islands_extended.indexing()

            for isl in islands_extended:
                unique_number_for_multiply += hash(isl[0])  # multiplayer

                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)
                isl.apply_aspect_ratio()

                # Constraints system
                ##################################################
                if found_univ_pro and self.bl_label == 'Unwrap' and self.constraints_weight and isl.has_constraints_edge():
                    uv = isl.umesh.uv
                    sync = isl.umesh.sync
                    for f in isl:
                        if f.select:
                            for crn in f.loops:
                                crn.tag = not crn[uv].pin_uv
                        else:
                            for crn in f.loops:
                                if crn[uv].pin_uv:
                                    crn.tag = False
                                else:
                                    linked = utils.linked_crn_to_vert_pair_with_seam(crn, uv, sync)
                                    crn.tag =  any(cc.face.select for cc in linked)

                    is_static = self.is_static_island(isl)
                    with utils.uv_parametrizer.unwrap_time_report(self.report):
                        failed_total += utils.uv_parametrizer.unwrap_isl_by_tag(isl,
                                                                            unwrap_along=self.unwrap_along,  # noqa
                                                                            use_abf=self.unwrap == 'ANGLE_BASED',
                                                                            topology_from_uvs=self.mark_seam_inner_island,
                                                                            blend_factor=self.blend_factor,
                                                                            fill_holes=self.fill_holes,
                                                                            constraints_factor=self.constraints_weight * 100)
                    hidden_constraints_islands.append(isl)
                    isl.reset_aspect_ratio()

                    if not is_static:
                        utils.set_global_texel(isl)
                    continue
                ##################################################

                self.unwrap_sync_faces_extend_select_and_set_pins(isl)

                save_t = isl.save_transform(flip_if_needed=True)
                save_t.save_coords(self.blend_factor)
                all_transform_islands.append(save_t)

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

        if all_transform_islands:
            for hidden_isl in hidden_constraints_islands:
                hidden_isl.sequence = hidden_isl.set_pins(with_pinned=True)

            bpy.ops.uv.unwrap(method=self.unwrap, fill_holes=self.fill_holes, correct_aspect=False, **unwrap_kwargs)

            for hidden_isl in hidden_constraints_islands:
                for crn_uv in hidden_isl.sequence:
                    crn_uv.pin_uv = False

        for isl in all_transform_islands:
            unpinned, to_select = isl.island.sequence
            for pin in unpinned:
                pin.pin_uv = False
            if isl.island.umesh.sync_valid:
                for crn in to_select:
                    crn.uv_select_vert = False
            else:
                for f in to_select:
                    f.select = False

            isl.inplace(flip_if_needed=False)
            isl.apply_saved_coords(self.blend_factor, flip_if_needed=True)
            isl.island.reset_aspect_ratio()

            if isl.rotate:
                utils.set_global_texel(isl.island)

        if failed_total:
            self.report({'WARNING'}, f"It is not possible to unwrap {failed_total!r} islands. "
                                     f"Try again by setting at least one pin or by partially selecting the island.")


    def unwrap_non_sync(self, **unwrap_kwargs):
        save_transform_islands = []
        hidden_constraints_islands = []
        failed_total = 0
        unique_number_for_multiply = 0
        tool_settings = bpy.context.scene.tool_settings
        is_sticky_mode_disabled = tool_settings.uv_sticky_select_mode == 'DISABLED'

        from importlib.util import find_spec
        found_univ_pro = find_spec(f"{__package__.rpartition('.')[0]}.univ_pro") is not None

        for umesh in reversed(self.umeshes):
            uv = umesh.uv
            face_select_get = utils.face_select_get_func(umesh)
            crn_select_get = utils.vert_select_get_func(umesh)

            umesh.value = umesh.check_uniform_scale(report=self.report)
            umesh.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
            islands = utypes.AdvIslands.calc_extended_any_elem_with_mark_seam(umesh)

            if not self.mark_seam_inner_island or umesh.bm.edges.layers.int.get('univ_constraints'):
                islands.indexing()

            for isl in islands:
                unique_number_for_multiply += hash(isl[0])  # multiplayer

                if self.mark_seam_inner_island:
                    isl.mark_seam(additional=True)
                else:
                    isl.mark_seam_by_index(additional=True)

                isl.apply_aspect_ratio()

                # Constraints system
                ##################################################
                if found_univ_pro and self.bl_label == 'Unwrap' and self.constraints_weight and isl.has_constraints_edge():
                    uv = isl.umesh.uv
                    sync = isl.umesh.sync
                    for f in isl:
                        if face_select_get(f):
                            for crn in f.loops:
                                crn.tag = not crn[uv].pin_uv
                        else:
                            for crn in f.loops:
                                if crn[uv].pin_uv:
                                    crn.tag = False
                                else:
                                    linked = utils.linked_crn_to_vert_pair_with_seam(crn, uv, sync)
                                    crn.tag = any(crn_select_get(cc) for cc in linked)

                    is_static = self.is_static_island(isl)

                    with utils.uv_parametrizer.unwrap_time_report(self.report):
                        failed_total += utils.uv_parametrizer.unwrap_isl_by_tag(isl,
                                                                            unwrap_along=self.unwrap_along,  # noqa
                                                                            use_abf=self.unwrap == 'ANGLE_BASED',
                                                                            topology_from_uvs=self.mark_seam_inner_island,
                                                                            blend_factor=self.blend_factor,
                                                                            fill_holes=self.fill_holes,
                                                                            constraints_factor=self.constraints_weight * 100)
                    hidden_constraints_islands.append(isl)
                    isl.reset_aspect_ratio()

                    if not is_static:
                        utils.set_global_texel(isl)
                    continue
                ##################################################
                else:
                    # Extend selection for avoid unlink unwrap
                    if is_sticky_mode_disabled:

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
                                    corners_to_select.add(crn)
                                else:
                                    temp_static.append(crn)
                            if has_selected:
                                for cc in temp_static:
                                    cc_uv = cc[uv]
                                    if not cc_uv.pin_uv:
                                        unpin_uvs.add(cc_uv)
                                        corners_to_select.add(cc)

                        for unpin_crn in unpin_uvs:
                            unpin_crn.pin_uv = True
                        if utils.USE_GENERIC_UV_SYNC:
                            for unsel_crn in corners_to_select:
                                unsel_crn.uv_select_vert = True
                        else:
                            for unsel_crn in corners_to_select:
                                unsel_crn[uv].select = True
                        isl.sequence = (unpin_uvs, corners_to_select)

                    save_t = isl.save_transform(flip_if_needed=True)
                    save_t.save_coords(self.blend_factor)
                    save_transform_islands.append(save_t)

        self.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

        if save_transform_islands:
            for hidden_isl in hidden_constraints_islands:
                hidden_isl.hide_for_unwrap()

            bpy.ops.uv.unwrap(method=self.unwrap, fill_holes=self.fill_holes, correct_aspect=False, **unwrap_kwargs)

            for hidden_isl in hidden_constraints_islands:
                hidden_isl.unhide_for_unwrap()

        for save_isl in save_transform_islands:
            save_isl.inplace(flip_if_needed=False)
            save_isl.apply_saved_coords(self.blend_factor, flip_if_needed=True)
            save_isl.island.reset_aspect_ratio()

            if save_isl.rotate:
                utils.set_global_texel(save_isl.island)

            if is_sticky_mode_disabled:
                if save_isl.island.sequence:
                    unpin_uvs, corners_to_select = save_isl.island.sequence
                    for unpin_crn in unpin_uvs:
                        unpin_crn.pin_uv = False

                    if utils.USE_GENERIC_UV_SYNC:
                        for unsel_crn in corners_to_select:
                            unsel_crn.uv_select_vert = False
                    else:
                        uv = save_isl.island.umesh.uv
                        for unsel_crn in corners_to_select:
                            unsel_crn[uv].select = False

        if failed_total:
            self.report({'WARNING'}, f"It is not possible to unwrap {failed_total!r} islands. "
                                     f"Try again by setting at least one pin or by partially selecting the island.")

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

    @staticmethod
    def is_static_island(isl):
        uv = isl.umesh.uv
        it = isl.corners_iter()
        for crn_a in it:
            if not crn_a.tag:
                co = crn_a[uv].uv
                for crn_b in it:
                    if not crn_b.tag:
                        if co != crn_b[uv].uv:
                            return True
                return False
        return False

    @staticmethod
    def lock_island_from_unwrap_and_get_pins_sync(isl):
        assert isl.umesh.sync
        uv = isl.umesh.uv
        pinned = []
        it = isl.corners_iter()
        if isl.umesh.sync_valid:
            for crn in it:
                if crn.uv_select_vert:
                    crn_uv = crn[uv]
                    if not crn_uv.pin_uv:
                        crn_uv.pin_uv = True
                        pinned.append(crn_uv)
        else:
            for crn in it:
                if crn.vert.select:
                    crn_uv = crn[uv]
                    if not crn_uv.pin_uv:
                        crn_uv.pin_uv = True
                        pinned.append(crn_uv)
        return pinned

# noinspection PyTypeHints
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

    fill_holes: bpy.props.BoolProperty(name='Fill Holes', default=True)
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(univ_settings(), 'use_texel')
        self.layout.prop(self, 'fill_holes')
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
            if not selected_umeshes:
                self.report({'WARNING'}, 'Need selected geometry')
                return {'CANCELLED'}

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
            mesh_island = hit.calc_mesh_island_with_seam()
            adv_subislands = mesh_island.calc_adv_subislands_with_mark_seam()
            for isl in adv_subislands:
                isl.select = True

            accidentally_selected_faces = UNIV_OT_Unwrap.prepare_accidentally_selected_islands_for_pick(mesh_island)
            for f in accidentally_selected_faces:
                f.hide = True

            unique_number_for_multiply = hash(mesh_island[0])  # multiplayer
            UNIV_OT_Unwrap.multiply_relax(unique_number_for_multiply, unwrap_kwargs)

            for isl in adv_subislands:
                isl.apply_aspect_ratio()
            save_t = utypes.SaveTransform(adv_subislands, flip_if_needed=True)

            bpy.ops.uv.unwrap(method=self.unwrap, fill_holes=self.fill_holes, correct_aspect=False, **unwrap_kwargs)

            adv_island = mesh_island.to_adv_island()
            save_t.island = adv_island
            save_t.inplace(flip_if_needed=True)

            adv_island.reset_aspect_ratio()

            if save_t.rotate:
                utils.set_global_texel(save_t.island)

            adv_island.select = False

            for f in accidentally_selected_faces:
                f.hide = False
                f.select = False
        else:
            mesh_island = hit.calc_mesh_island_with_seam()
            adv_island = mesh_island.to_adv_island()
            adv_island.select = True

            bpy.ops.uv.unwrap(method=self.unwrap, fill_holes=self.fill_holes, correct_aspect=False, **unwrap_kwargs)

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
                        unique_number += self.unwrap_selected_verts_preprocess(umesh)
                elif self.umeshes.elem_mode == 'EDGE':
                    if umesh.total_face_sel:
                        unique_number += self.unwrap_selected_faces_preprocess_vert_edge_mode(umesh)
                    else:
                        unique_number += self.unwrap_selected_edges_preprocess(umesh)
                else:
                    unique_number += self.unwrap_selected_faces_preprocess(umesh)

        UNIV_OT_Unwrap.multiply_relax(unique_number % (1 << 62), unwrap_kwargs)
        bpy.ops.uv.unwrap(method=self.unwrap, correct_aspect=False, **unwrap_kwargs)

        for umesh in meshes_with_uvs:
            self.unwrap_selected_faces_postprocess(umesh)
            umesh.bm.select_flush(False)

        self.unwrap_without_uvs_postprocess(meshes_without_uvs)

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
            safe_transform = utypes.SaveTransform(adv_islands, flip_if_needed=True)
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

    def unwrap_selected_verts_preprocess(self, umesh):
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
            safe_transform = utypes.SaveTransform(adv_islands, flip_if_needed=True)
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

    def unwrap_selected_edges_preprocess(self, umesh):
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
            safe_transform = utypes.SaveTransform(adv_islands, flip_if_needed=True)
            save_transform_islands.append(safe_transform)

            to_select.extend(mesh_isl)

            for f in mesh_isl:
                # Pin unselected faces
                if all(not v.select for v in f.verts):
                    for crn in f.loops:
                        crn_uv = crn[uv]
                        if crn_uv.pin_uv:
                            continue
                        crn_uv.pin_uv = True
                        pinned.append(crn_uv)
                    continue

                # Pin partial selected
                for crn in f.loops:
                    if crn.edge.select:
                        continue
                    crn_uv = crn[uv]
                    if crn_uv.pin_uv:
                        continue

                    if crn.vert.select:
                        if any(crn_.edge.select for crn_ in utils.linked_crn_to_vert_with_seam_3d_iter(crn)):
                            continue
                        # Over check after linked_crn_to_vert_with_seam_3d_iter,
                        # because it returns nothing for vertices without linked faces (see note).
                        if crn.link_loop_prev.edge.select:
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
            safe_transform = utypes.SaveTransform(adv_islands, flip_if_needed=True)
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
            safe_transform.inplace_mesh_island(flip_if_needed=True)
            safe_transform.island.reset_aspect_ratio()

            if safe_transform.rotate:
                if isinstance(safe_transform.island, utypes.AdvIsland):
                    utils.set_global_texel(safe_transform.island)
                else:
                    for isl in safe_transform.island:
                        utils.set_global_texel(isl)

    def unwrap_without_uvs_postprocess(self, umeshes):
        for umesh in umeshes:
            mesh_islands = utypes.MeshIslands.calc_extended_with_mark_seam(umesh)
            umesh.verify_uv()

            adv_islands = mesh_islands.to_adv_islands()

            adv_islands.calc_area_uv()
            adv_islands.calc_area_3d(scale=umesh.value)

            # reset aspect
            for adv_isl in adv_islands:
                adv_isl.set_texel(self.texel, self.texture_size)
                if umesh.aspect != 1.0:
                    scale = Vector((1 / umesh.aspect, 1))
                    adv_isl.scale(scale, adv_isl.bbox.center)
