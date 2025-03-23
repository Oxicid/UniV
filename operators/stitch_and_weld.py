# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
from itertools import chain
from mathutils import Vector

from bmesh.types import BMLoop
from bpy.types import Operator
from bpy.props import *

from .. import utils
from .. import types
from ..types import (
    BBox,
    UMeshes,
    Islands,
    AdvIslands,
    AdvIsland,
    LoopGroup
)
from ..preferences import prefs, univ_settings


class UNIV_OT_Weld(Operator):
    bl_idname = "uv.univ_weld"
    bl_label = "Weld"
    bl_description = "Weld selected UV vertices\n\n" \
                     "If there are paired and unpaired selections with no connections \nat the same time in the off sync mode, \n" \
                     "the paired connection is given priority, but when you press again, \nthe unpaired selections are also connected.\n" \
                     "This prevents unwanted connections.\n" \
                     "Works like Stitch if everything is welded in the island.\n\n" \
                     "Context keymaps on button:\n" \
                     "Default - Weld\n" \
                     "Alt - Weld by Distance\n\n" \
                     "Has [W] keymap"
    bl_options = {'REGISTER', 'UNDO'}

    use_by_distance: BoolProperty(name='By Distance', default=False)
    distance: FloatProperty(name='Distance', default=0.0005, min=0, soft_max=0.05, step=0.0001)  # noqa
    weld_by_distance_type: EnumProperty(name='Weld by', default='BY_ISLANDS', items=(
        ('ALL', 'All', ''),
        ('BY_ISLANDS', 'By Islands', '')
    ))

    flip: BoolProperty(name='Flip', default=False, options={'HIDDEN'})
    use_aspect: BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        if self.use_by_distance:
            layout.row(align=True).prop(self, 'weld_by_distance_type', expand=True)
        row = layout.row(align=True)
        row.prop(self, "use_by_distance", text="")
        row.active = self.use_by_distance
        row.prop(self, 'distance', slider=True)
        layout.prop(self, 'use_aspect')

    def invoke(self, context, event):
        if event.value == 'PRESS':
            if context.area.ui_type == 'UV':
                self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
                self.mouse_position = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)
        self.use_by_distance = event.alt

        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync = bpy.context.scene.tool_settings.use_uv_select_sync
        self.umeshes: UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_position: Vector | None = None
        self.update_seams = True

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio() if self.use_aspect else 1.0

        if self.use_by_distance:
            selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_verts()
            self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

            if not self.umeshes:
                return self.umeshes.update()

            if self.weld_by_distance_type == 'BY_ISLANDS':
                self.weld_by_distance_island(extended=bool(selected_umeshes))
            else:
                self.weld_by_distance_all(selected=bool(selected_umeshes))
        else:
            selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()
            self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

            if not self.umeshes:
                return self.umeshes.update()

            from .. import draw
            for umesh in chain(selected_umeshes, visible_umeshes):
                umesh.sequence = draw.mesh_extract.extract_edges_with_seams(umesh)

            if not selected_umeshes and self.mouse_position:
                self.pick_weld()
                self.filter_and_draw_lines(selected_umeshes, visible_umeshes)
                bpy.context.area.tag_redraw()
                return {'FINISHED'}

            self.weld()
            self.filter_and_draw_lines(selected_umeshes, visible_umeshes)
            bpy.context.area.tag_redraw()

        self.umeshes.update(info='Not found verts for weld')
        return {'FINISHED'}

    def weld(self):
        from ..utils import weld_crn_edge_by_idx

        islands_of_mesh = []
        for umesh in self.umeshes:
            uv = umesh.uv
            update_tag = False
            if islands := Islands.calc_extended_any_edge_non_manifold(umesh):
                umesh.set_corners_tag(False)
                islands.indexing()

                for idx, isl in enumerate(islands):
                    isl.set_selected_crn_edge_tag(umesh)

                    for crn in isl.iter_corners_by_tag():
                        shared = crn.link_loop_radial_prev
                        if shared == crn:
                            crn.tag = False
                            continue

                        if shared.face.index != idx:  # island boundary skip
                            crn.tag = False
                            shared.tag = False
                            continue

                        if not shared.tag:  # single select preserve system
                            continue

                        # CPU Bound
                        crn_next = crn.link_loop_next
                        shared_next = shared.link_loop_next

                        is_splitted_a = crn[uv].uv != shared_next[uv].uv
                        is_splitted_b = crn_next[uv].uv != shared[uv].uv

                        if is_splitted_a and is_splitted_b:
                            weld_crn_edge_by_idx(crn, shared_next, idx, uv)
                            weld_crn_edge_by_idx(crn_next, shared, idx, uv)
                            update_tag |= True
                        elif is_splitted_a:
                            weld_crn_edge_by_idx(crn, shared_next, idx, uv)
                            update_tag |= True
                        elif is_splitted_b:
                            weld_crn_edge_by_idx(crn_next, shared, idx, uv)
                            update_tag |= True

                        edge = crn.edge
                        if edge.seam:
                            edge.seam = False
                            update_tag |= True

                        crn.tag = False
                        shared.tag = False
            umesh.update_tag = update_tag

            if islands:
                islands_of_mesh.append(islands)

        if self.umeshes.update_tag:
            return

        if not self.umeshes.sync:
            for islands in islands_of_mesh:
                update_tag = False
                uv = islands.umesh.uv
                for idx, isl in enumerate(islands):
                    for crn in isl.iter_corners_by_tag():
                        utils.copy_pos_to_target_with_select(crn, uv, idx)
                        if crn.edge.seam:
                            crn.edge.seam = False
                        update_tag |= True
                islands.umesh.update_tag = update_tag

            if self.umeshes.update_tag:
                return

        UNIV_OT_Stitch.stitch(self)  # noqa TODO: Implement inheritance

    def weld_by_distance_island(self, extended):
        for umesh in self.umeshes:
            uv = umesh.uv
            update_tag = False
            if islands := Islands.calc_any_extended_or_visible_non_manifold(umesh, extended=extended):
                # Tagging
                for f in umesh.bm.faces:
                    for crn in f.loops:
                        crn.tag = False
                for isl in islands:
                    if extended:
                        isl.tag_selected_corner_verts_by_verts(umesh)
                    else:
                        isl.set_corners_tag(True)

                    for crn in isl.iter_corners_by_tag():
                        crn_in_vert = [crn_v for crn_v in crn.vert.link_loops if crn_v.tag]
                        update_tag |= self.weld_corners_in_vert(crn_in_vert, uv)

                if update_tag:
                    for isl in islands:
                        isl.mark_seam()
            umesh.update_tag = update_tag

        if not self.umeshes.update_tag:
            self.umeshes.cancel_with_report(info='Not found verts for weld')

    def weld_by_distance_all(self, selected):
        # TODO: Refactor this, use iterator
        for umesh in self.umeshes:
            umesh.tag_visible_corners()
            uv = umesh.uv
            update_tag = False
            if selected:
                init_corners = utils.calc_selected_uv_vert_corners(umesh)
            else:
                init_corners = utils.calc_visible_uv_corners(umesh)
            if init_corners:
                # Tagging
                is_face_mesh_mode = (self.sync and utils.get_select_mode_mesh() == 'FACE')
                if not is_face_mesh_mode:
                    for f in umesh.bm.faces:
                        for crn in f.loops:
                            crn.tag = False

                for crn in init_corners:
                    crn.tag = True

                if is_face_mesh_mode:
                    if selected:
                        for f in umesh.bm.faces:
                            for crn in f.loops:
                                if not crn.face.select:
                                    crn.tag = False

                corners = (crn for crn in init_corners if crn.tag)
                for crn in corners:
                    crn_in_vert = [crn_v for crn_v in crn.vert.link_loops if crn_v.tag]
                    self.weld_corners_in_vert(crn_in_vert, uv)  # update_tag |=

            if init_corners:  # TODO: Count deleted seams and weld_corners_in_vert
                umesh.tag_visible_faces()
                umesh.mark_seam_tagged_faces()

            umesh.update_tag = update_tag

        if not self.umeshes.update_tag:
            self.umeshes.cancel_with_report(info='Not found verts for weld')

    def weld_corners_in_vert(self, crn_in_vert, uv):
        if utils.all_equal(_crn[uv].uv for _crn in crn_in_vert):
            for crn_t in crn_in_vert:
                crn_t.tag = False
            return False

        for group in self.calc_distance_groups(crn_in_vert, uv):
            value = Vector((0, 0))
            for c in group:
                value += c[uv].uv
            avg = value / len(group)
            for c in group:
                c[uv].uv = avg
        return True

    def calc_distance_groups(self, crn_in_vert: list[BMLoop], uv) -> list[list[BMLoop]]:
        corners_groups = []
        union_corners = []
        for corner_first in crn_in_vert:
            if not corner_first.tag:
                continue
            corner_first.tag = False

            union_corners.append(corner_first)
            compare_index = 0
            while True:
                if compare_index > len(union_corners) - 1:
                    if utils.all_equal(_crn[uv].uv for _crn in union_corners):
                        union_corners = []
                        break
                    corners_groups.append(union_corners)
                    union_corners = []
                    break

                for isl in crn_in_vert:
                    if not isl.tag:
                        continue

                    if (union_corners[compare_index][uv].uv - isl[uv].uv).length <= self.distance:
                        isl.tag = False
                        union_corners.append(isl)
                compare_index += 1
        return corners_groups

    def pick_weld(self):
        hit = types.CrnEdgeHit(self.mouse_position, self.max_distance)
        for umesh in self.umeshes:
            hit.find_nearest_crn_by_visible_faces(umesh)

        if not hit:
            self.report({'WARNING'}, 'Edge not found within a given radius')
            return

        sync = hit.umesh.sync
        ref_crn = hit.crn
        shared = ref_crn.link_loop_radial_prev
        is_visible = utils.is_visible_func(sync)
        if shared == ref_crn or not is_visible(shared.face):
            self.report({'INFO'}, 'Edge is boundary')
            return

        if ref_crn.link_loop_next.vert != shared.vert:
            self.report({'WARNING'}, 'Edge has 3D flipped face, need recalculate normals')
            return

        uv = hit.umesh.uv
        e = ref_crn.edge
        if utils.is_pair(ref_crn, shared, uv):
            if e.seam:
                e.seam = False
                hit.umesh.update()
                return
            return

        # fast calculate, if edge has non-manifold links
        if ref_crn[uv].uv == shared.link_loop_next[uv].uv:  # check a
            for crn in [shared] + utils.linked_crn_to_vert_pair_with_seam(shared, uv, sync):
                crn[uv].uv = ref_crn.link_loop_next[uv].uv  # link b
            e.seam = False
            hit.umesh.update()
            return
        elif ref_crn.link_loop_next[uv].uv == shared[uv].uv:  # check b
            shared_next_crn = shared.link_loop_next
            for crn in [shared_next_crn] + utils.linked_crn_to_vert_pair_with_seam(shared_next_crn, uv, sync):
                crn[uv].uv = ref_crn[uv].uv  # link a
            e.seam = False
            hit.umesh.update()
            return

        ref_isl, isl_set = hit.calc_island_non_manifold()

        if shared.face in isl_set:
            shared_next_crn = shared.link_loop_next
            for crn in [shared_next_crn] + utils.linked_crn_to_vert_pair_with_seam(shared_next_crn, uv, sync):
                crn[uv].uv = ref_crn[uv].uv

            for crn in [shared] + utils.linked_crn_to_vert_pair_with_seam(shared, uv, sync):
                crn[uv].uv = ref_crn.link_loop_next[uv].uv
            e.seam = False
            hit.umesh.update()
        else:
            hit.crn = shared
            trans_isl, _ = hit.calc_island_with_seam()
            UNIV_OT_Stitch.stitch_pick_ex(ref_isl, trans_isl, ref_crn, shared)
            hit.umesh.update()
        return

    @staticmethod
    def filter_and_draw_lines(umeshes_a, umeshes_b):
        welded = []
        with_seam = []
        welded_append = welded.append
        with_seam_append = with_seam.append

        is_visible = utils.is_visible_func(umeshes_a.sync)

        # TODO: Optimize by update tag umesh (if umesh without tag_update -> not welded edges)
        for umesh in chain(umeshes_a, umeshes_b):
            uv = umesh.uv
            for e in umesh.sequence:
                if e.seam:
                    for crn in e.link_loops:
                        if is_visible(crn.face):
                            with_seam_append(crn[uv].uv)
                            with_seam_append(crn.link_loop_next[uv].uv)
                else:
                    toggle = False
                    for crn in e.link_loops:
                        if is_visible(crn.face):
                            if toggle:
                                welded_append(crn[uv].uv)
                                welded_append(crn.link_loop_next[uv].uv)
                            else:
                                welded_append(crn.link_loop_next[uv].uv)
                                welded_append(crn[uv].uv)
                                toggle = True

        from .. import draw
        seam_color = (*bpy.context.preferences.themes[0].view_3d.edge_seam, 0.8)
        draw.LinesDrawSimple.draw_register(with_seam, seam_color)

        welded_color = (0.1, 0.1, 1.0, 1.0)
        draw.DotLinesDrawSimple.draw_register(welded, welded_color)


class UNIV_OT_Stitch(Operator):
    bl_idname = "uv.univ_stitch"
    bl_label = 'Stitch'
    bl_description = "Stitch selected UV vertices by proximity\n\n" \
                     "Default - Stitch\n" \
                     "Alt - Stitch Between\n\n" \
                     "Has [Shift + W] keymap. \n" \
                     "In sync mode when calling stitch via keymap, the stitch priority is done by mouse cursor.\n" \
                     "In other cases of pairwise selection, prioritization occurs by island size"
    bl_options = {'REGISTER', 'UNDO'}

    between: BoolProperty(name='Between', default=False, description='Attention, it is unstable')
    update_seams: BoolProperty(name='Update Seams', default=True)
    flip: BoolProperty(name='Flip', default=False)
    padding_multiplayer: FloatProperty(name='Padding Multiplayer', default=0.0, min=0, soft_max=1, max=4)
    use_aspect: BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "between")
        layout.prop(self, "update_seams")
        layout.prop(self, "flip")
        layout.prop(self, "use_aspect")

        if self.padding_multiplayer:
            layout.separator(factor=0.35)
            settings = univ_settings()
            layout.label(text=f"Global Texture Size = {settings.size_x}x{settings.size_y}")
            layout.label(text=f"Padding = {settings.padding}({int(settings.padding * self.padding_multiplayer)})px")
        layout.prop(self, "padding_multiplayer", slider=True)


    def invoke(self, context, event):
        if event.value == 'PRESS':
            if context.area.ui_type == 'UV':
                self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
                self.mouse_position = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)
        self.between = event.alt
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync = utils.sync()
        self.umeshes: UMeshes | None = None
        self.max_distance: float = 0.0
        self.mouse_position: Vector | None = None
        self.padding = 0.0

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        settings = univ_settings()
        self.padding = int(settings.padding * self.padding_multiplayer) / min(int(settings.size_x), int(settings.size_y))

        from .. import draw
        for umesh in chain(selected_umeshes, visible_umeshes):
            umesh.aspect = utils.get_aspect_ratio() if self.use_aspect else 1.0
            umesh.sequence = draw.mesh_extract.extract_edges_with_seams(umesh)

        if self.between:
            if not selected_umeshes:
                self.umeshes.umeshes = []
            self.stitch_between()
        else:
            if not self.umeshes:
                return self.umeshes.update()
            if not selected_umeshes and self.mouse_position:
                if self.padding and (img_size := utils.get_active_image_size()):
                    if (int(settings.size_x), int(settings.size_y)) != img_size:
                        self.report({'WARNING'}, 'Global and Active texture sizes have different values, '
                                                 'which will result in incorrect padding.')

                self.pick_stitch()
                UNIV_OT_Weld.filter_and_draw_lines(selected_umeshes, visible_umeshes)
                bpy.context.area.tag_redraw()
                return {'FINISHED'}

            self.stitch()

        UNIV_OT_Weld.filter_and_draw_lines(selected_umeshes, visible_umeshes)
        bpy.context.area.tag_redraw()

        # TODO: Remove seams from selected edges (self.stitch and self.stitch_between)
        self.umeshes.update(info='Not found edges for stitch')
        return {'FINISHED'}

    def stitch(self):
        for umesh in self.umeshes:
            if self.sync and umesh.is_full_edge_deselected or (not self.sync and umesh.is_full_face_deselected):
                umesh.update_tag = False
                continue

            uv = umesh.uv
            adv_islands = AdvIslands.calc_extended_or_visible(umesh, extended=False)
            if len(adv_islands) < 2:
                umesh.update_tag = False
                continue

            adv_islands.indexing()

            for f in umesh.bm.faces:
                for crn in f.loops:
                    crn.tag = False

            if self.sync:
                target_islands = [isl for isl in adv_islands if any(crn.edge.select for f in isl for crn in f.loops)]
            else:
                target_islands = [isl for isl in adv_islands if any(crn[uv].select_edge for f in isl for crn in f.loops)]

            if self.sync and self.mouse_position:
                def sort_by_nearest_to_mouse(__island: AdvIsland) -> float:
                    nonlocal uv
                    nonlocal mouse_position

                    min_dist = math.inf
                    for _ff in __island:
                        for crn_ in _ff.loops:
                            if crn_.edge.select:
                                closest_pt = utils.closest_pt_to_line(mouse_position, crn_[uv].uv, crn_.link_loop_next[uv].uv)
                                min_dist = min(min_dist, (closest_pt-mouse_position).length_squared)
                    return min_dist

                mouse_position = self.mouse_position
                target_islands.sort(key=sort_by_nearest_to_mouse)

            else:
                for _isl in target_islands:
                    _isl.calc_selected_edge_length()

                target_islands.sort(key=lambda a: a.info.edge_length, reverse=True)  # TODO: Replace info

                for _isl in reversed(target_islands):
                    if _isl.info.edge_length < 1e-06:
                        target_islands.remove(_isl)

            if not target_islands:
                umesh.update_tag = False
                continue

            update_tag = False
            while True:
                stitched = False
                for target_isl in target_islands:
                    tar = LoopGroup(umesh)

                    while True:
                        local_stitched = False
                        if target_isl:  # TODO: Stitch_ex remove faces and other attrs, change logic
                            for _ in tar.calc_first(target_isl):
                                source = tar.calc_shared_group()
                                res = UNIV_OT_Stitch.stitch_ex(self, tar, source, adv_islands)
                                local_stitched |= res
                        stitched |= local_stitched
                        if not local_stitched:
                            break
                update_tag |= stitched
                if not stitched:
                    break

            if update_tag and self.update_seams:
                for adv in adv_islands:
                    if adv:
                        adv.mark_seam()
            umesh.update_tag = update_tag

    def stitch_between(self):
        for umesh in self.umeshes:
            if umesh.is_full_face_deselected:
                umesh.update_tag = False
                continue
            uv = umesh.uv
            _islands = AdvIslands.calc_extended_or_visible(umesh, extended=True)
            if len(_islands) < 2:
                umesh.update_tag = False
                continue

            _islands.indexing()

            for f in umesh.bm.faces:
                for crn in f.loops:
                    crn.tag = False

            target_islands = _islands.islands[:]
            if self.sync and self.mouse_position:
                def sort_by_nearest_to_mouse(__island: AdvIsland) -> float:
                    from mathutils.geometry import intersect_point_line
                    nonlocal uv
                    nonlocal mouse_position

                    min_dist = math.inf
                    for _ff in __island:
                        for __crn_sort in _ff.loops:
                            intersect = intersect_point_line(mouse_position, __crn_sort[uv].uv, __crn_sort.link_loop_next[uv].uv)
                            dist = (mouse_position - intersect[0]).length
                            if min_dist > dist:
                                min_dist = dist
                    return min_dist

                mouse_position = self.mouse_position
                target_islands.sort(key=sort_by_nearest_to_mouse)

            else:
                for _isl in target_islands:
                    _isl.calc_selected_edge_length(selected=False)

                target_islands.sort(key=lambda a: a.info.edge_length, reverse=True)

                for _isl in reversed(target_islands):
                    if _isl.info.edge_length < 1e-06:
                        target_islands.remove(_isl)

            if not target_islands:
                umesh.update_tag = False
                continue

            update_tag = False
            while True:
                stitched = False
                for target_isl in target_islands:
                    tar = LoopGroup(umesh)

                    while True:
                        local_stitched = False
                        for _ in tar.calc_first(target_isl, selected=False):
                            source = tar.calc_shared_group()
                            res = self.stitch_ex(tar, source, _islands, selected=False)
                            local_stitched |= res
                        stitched |= local_stitched
                        if not local_stitched:
                            break
                update_tag |= stitched
                if not stitched:
                    break
            if update_tag and self.update_seams:
                for adv in target_islands:
                    if adv:
                        adv.mark_seam()
            umesh.update_tag = update_tag

    @staticmethod
    def has_zero_length(crn_a1, crn_a2, crn_b1, crn_b2, uv):
        return (crn_a1[uv].uv - crn_a2[uv].uv).length < 1e-06 or \
            (crn_b1[uv].uv - crn_b2[uv].uv).length < 1e-06

    @staticmethod
    def calc_begin_end_points(tar: LoopGroup, source: LoopGroup):
        if not tar or not source:
            return False
        uv = tar.umesh.uv

        crn_a1 = tar[0]
        crn_a2 = tar[-1].link_loop_next
        crn_b1 = source[-1].link_loop_next
        crn_b2 = source[0]

        # If zero length LoopGroup might be circular
        if UNIV_OT_Stitch.has_zero_length(crn_a1, crn_a2, crn_b1, crn_b2, uv):
            bbox, bbox_margin_corners = BBox.calc_bbox_with_corners(tar, tar.umesh.uv)
            xmin_crn, xmax_crn, ymin_crn, ymax_crn = bbox_margin_corners
            if bbox.max_length < 1e-06:
                return False

            if bbox.width > bbox.height:
                crn_a1 = xmin_crn
                crn_a2 = xmax_crn

                crn_b1 = utils.shared_crn(xmin_crn).link_loop_next
                crn_b2 = utils.shared_crn(xmax_crn).link_loop_next
            else:
                crn_a1 = ymin_crn
                crn_a2 = ymax_crn

                crn_b1 = utils.shared_crn(ymin_crn).link_loop_next
                crn_b2 = utils.shared_crn(ymax_crn).link_loop_next

            if UNIV_OT_Stitch.has_zero_length(crn_a1, crn_a2, crn_b1, crn_b2, uv):
                return False

        return crn_a1, crn_a2, crn_b1, crn_b2

    @staticmethod
    def copy_pos(crn, uv):
        co_a = crn[uv].uv
        shared_a = utils.shared_crn(crn).link_loop_next
        source_corners = utils.linked_crn_uv_by_face_index(shared_a, uv)
        for _crn in source_corners:
            _crn[uv].uv = co_a

        co_b = crn.link_loop_next[uv].uv
        shared_b = utils.shared_crn(crn)
        source_corners = utils.linked_crn_uv_by_face_index(shared_b, uv)
        for _crn in source_corners:
            _crn[uv].uv = co_b

    def stitch_ex(self, ref_lg: LoopGroup, trans_lg: LoopGroup, adv_islands: AdvIslands, selected=True):
        uv = ref_lg.umesh.uv
        # Equal indices occur after merging on non-stitch edges
        # TODO: Disable?
        if ref_lg[0].face.index == trans_lg[0].face.index:
            for target_crn in ref_lg:
                UNIV_OT_Stitch.copy_pos(target_crn, uv)
            return True

        if (corners := UNIV_OT_Stitch.calc_begin_end_points(ref_lg, trans_lg)) is False:
            ref_lg.tag = False
            return False

        ref_isl = adv_islands[corners[0].face.index]
        trans_isl = adv_islands[corners[2].face.index]

        # if not (ref_isl.is_full_flipped != trans_isl.is_full_flipped):  # TODO: Implement auto flip (reverse LoopGroup?)
        #     trans_isl.scale_simple(Vector((1, -1)))

        if self.flip:
            trans_isl.scale_simple(Vector((1, -1)))
        pt_a1, pt_a2, pt_b1, pt_b2 = [c[uv].uv for c in corners]

        # Rotate
        normal_a_with_aspect = (pt_a1 - pt_a2) * Vector((ref_lg.umesh.aspect, 1.0))
        normal_b_with_aspect = (pt_b1 - pt_b2) * Vector((ref_lg.umesh.aspect, 1.0))

        rotate_angle = normal_a_with_aspect.angle_signed(normal_b_with_aspect, 0)
        trans_isl.rotate_simple(rotate_angle, ref_lg.umesh.aspect)

        # Scale
        normal_a = pt_a1 - pt_a2
        normal_b = pt_b1 - pt_b2

        scale = normal_a.length / normal_b.length
        trans_isl.scale_simple(Vector((scale, scale)))

        # Move
        trans_isl.move(pt_a1 - pt_b1)

        adv_islands.weld_selected(ref_isl, trans_isl, selected=selected)
        return True

    @staticmethod
    def stitch_pick_ex(ref_isl: AdvIsland,
                       trans: AdvIsland,
                       ref_crn: BMLoop,
                       trans_crn: BMLoop,
                       update_seams=True,
                       pad=0.0):
        uv = ref_isl.umesh.uv

        if utils.is_flipped_uv(ref_crn.face, uv) != utils.is_flipped_uv(trans_crn.face, uv):
            trans.scale_simple(Vector((1, -1)))

        pt_a1, pt_a2 = ref_crn[uv].uv, ref_crn.link_loop_next[uv].uv
        pt_b1, pt_b2 = trans_crn.link_loop_next[uv].uv, trans_crn[uv].uv

        # Rotate
        normal_a_with_aspect = (pt_a1 - pt_a2) * Vector((ref_isl.umesh.aspect, 1.0))
        normal_b_with_aspect = (pt_b1 - pt_b2) * Vector((ref_isl.umesh.aspect, 1.0))

        rotate_angle = normal_a_with_aspect.angle_signed(normal_b_with_aspect, 0)
        trans.rotate_simple(rotate_angle, ref_isl.umesh.aspect)

        # Scale
        normal_a = pt_a1 - pt_a2
        normal_b = pt_b1 - pt_b2
        length_a = normal_a.length
        length_b = normal_b.length

        if not (length_a < 1e-06 or length_b < 1e-06):
            scale = length_a / length_b
            trans.scale_simple(Vector((scale, scale)))

        # Move
        if pad:
            aspect_vec = Vector((1 / ref_isl.umesh.aspect, 1))
            orto = normal_a.orthogonal().normalized() * pad
            if orto == Vector((0, 0)):  # TODO: Convert static to default method and report zero length
                orto = (trans.bbox.center - ref_isl.bbox.center) * Vector((ref_isl.umesh.aspect, 1.0))
                orto = orto.normalized() * pad
            if orto == Vector((0, 0)):
                orto = Vector((pad, 0))
            orto *= aspect_vec
            delta = (pt_a1 - pt_b1) + orto
            trans.move(delta)
        else:
            trans.move(pt_a1 - pt_b1)

            trans_a = [trans_crn] + utils.linked_crn_to_vert_pair_with_seam(trans_crn, uv, ref_isl.umesh.sync)
            for crn in trans_a:
                crn[uv].uv = ref_crn.link_loop_next[uv].uv

            trans_next = trans_crn.link_loop_next
            trans_b = [trans_next] + utils.linked_crn_to_vert_pair_with_seam(trans_next, uv, ref_isl.umesh.sync)
            for crn in trans_b:
                crn[uv].uv = ref_crn[uv].uv

            if update_seams:
                ref_crn.edge.seam = False

    def pick_stitch(self):
        hit = types.CrnEdgeHit(self.mouse_position, self.max_distance)
        for umesh in self.umeshes:
            hit.find_nearest_crn_by_visible_faces(umesh)

        if not hit:
            self.report({'WARNING'}, 'Edge not found within a given radius')
            return

        sync = hit.umesh.sync
        ref_crn = hit.crn
        shared = ref_crn.link_loop_radial_prev
        is_visible = utils.is_visible_func(sync)
        if shared == ref_crn or not is_visible(shared.face):
            self.report({'WARNING'}, 'Edge is boundary')
            return

        if ref_crn.link_loop_next.vert != shared.vert:
            self.report({'WARNING'}, 'Edge has 3D flipped face, need recalculate normals')
            return

        uv = hit.umesh.uv
        e = ref_crn.edge
        if not self.padding and utils.is_pair(ref_crn, shared, uv):
            if e.seam:
                e.seam = False
                hit.umesh.update()
            else:
                self.report({'INFO'}, 'The edge was already stitched, no action was taken')
            return

        ref_isl, isl_set = hit.calc_island_with_seam()

        if shared.face in isl_set:
            self.report({'WARNING'}, 'It is not possible to use Stitch on itself, use Weld operator')
        else:
            hit.crn = shared
            trans_isl, _ = hit.calc_island_with_seam()
            self.stitch_pick_ex(ref_isl, trans_isl, ref_crn, shared, self.update_seams, self.padding)
            hit.umesh.update()
