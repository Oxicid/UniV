# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import random
import bl_math
import typing  # noqa
from collections.abc import Callable

import numpy as np

from bpy.types import Operator
from bpy.props import *

from math import pi, sin, cos, atan2, isclose, radians as to_rad
from mathutils import Vector
from collections import defaultdict

from .. import utils
from .. import types
from .. import info
from ..types import (
    BBox,
    UMeshes,
    Islands,
    AdvIslands,
    AdvIsland,
    FaceIsland,
    UnionIslands
)
from ..preferences import prefs, univ_settings


class UNIV_OT_Crop(Operator, utils.PaddingHelper):
    bl_idname = 'uv.univ_crop'
    bl_label = 'Crop'
    bl_description = info.operator.crop_info
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(name='Axis', default='XY', items=(('XY', 'Both', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    to_cursor: BoolProperty(name='To Cursor', default=False)
    individual: BoolProperty(name='Individual', default=False)
    inplace: BoolProperty(name='Inplace', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        layout.row(align=True).prop(self, 'axis', expand=True)
        layout.prop(self, 'to_cursor')
        layout.prop(self, 'individual')
        if not self.to_cursor:
            layout.prop(self, 'inplace')
        self.draw_padding()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode: str = 'DEFAULT'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.to_cursor = event.ctrl
        self.individual = event.shift
        self.inplace = event.alt

        if all((event.ctrl, event.alt)):
            self.report({'INFO'}, f"Event: {utils.event_to_string(event)} not implement. \n\n"
                                  f"See all variations:\n\n{self.get_event_info()}")
            self.to_cursor = False
        return self.execute(context)

    def execute(self, context):
        self.mode_preprocessing()
        self.calc_padding()
        self.report_padding()
        return self.crop(self.mode, self.axis, self.padding, proportional=True, report=self.report)

    def mode_preprocessing(self):
        if all((self.to_cursor, self.inplace)):
            self.inplace = False
        match self.to_cursor, self.individual, self.inplace:
            case False, False, False:
                self.mode = 'DEFAULT'
            case True, False, False:
                self.mode = 'TO_CURSOR'
            case True, True, False:
                self.mode = 'TO_CURSOR_INDIVIDUAL'
            case False, True, False:
                self.mode = 'INDIVIDUAL'
            case False, False, True:
                self.mode = 'INPLACE'
            case False, True, True:
                self.mode = 'INDIVIDUAL_INPLACE'

    @staticmethod
    def crop(mode, axis, padding, proportional, report=None):
        umeshes = types.UMeshes(report=report)
        crop_args = [axis, padding, umeshes, proportional]

        match mode:
            case 'DEFAULT':
                UNIV_OT_Crop.crop_default(*crop_args, extended=True)
                if not umeshes.final():
                    UNIV_OT_Crop.crop_default(*crop_args, extended=False)
            case 'TO_CURSOR':
                if not (offset := utils.get_tile_from_cursor()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Crop.crop_default(*crop_args, offset=offset, extended=True)
                if not umeshes.final():
                    UNIV_OT_Crop.crop_default(*crop_args, offset=offset, extended=False)
            case 'TO_CURSOR_INDIVIDUAL':
                if not (offset := utils.get_tile_from_cursor()):
                    if report:
                        report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Crop.crop_individual(*crop_args, offset=offset, extended=True)
                if not umeshes.final():
                    UNIV_OT_Crop.crop_individual(*crop_args, offset=offset, extended=False)
            case 'INDIVIDUAL':
                UNIV_OT_Crop.crop_individual(*crop_args, extended=True)
                if not umeshes.final():
                    UNIV_OT_Crop.crop_individual(*crop_args, extended=False)
            case 'INDIVIDUAL_INPLACE':
                UNIV_OT_Crop.crop_individual(*crop_args, inplace=True, extended=True)
                if not umeshes.final():
                    UNIV_OT_Crop.crop_individual(*crop_args, inplace=True, extended=False)
            case 'INPLACE':
                UNIV_OT_Crop.crop_inplace(*crop_args, extended=True)
                if not umeshes.final():
                    UNIV_OT_Crop.crop_inplace(*crop_args, extended=False)
            case _:
                raise NotImplementedError(mode)

        return umeshes.update()

    @staticmethod
    def crop_default(axis, padding, umeshes, proportional, offset=Vector((0, 0)), inplace=False, extended=True):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
            umesh.update_tag = bool(islands)

        if not islands_of_mesh:
            return

        UNIV_OT_Crop.crop_ex(axis, general_bbox, inplace, islands_of_mesh, offset, padding, proportional)

    @staticmethod
    def crop_individual(axis, padding, umeshes, proportional, offset=Vector((0, 0)), inplace=False, extended=True):
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh, extended=extended):
                for island in islands:
                    UNIV_OT_Crop.crop_ex(axis, island.calc_bbox(), inplace, (island, ), offset, padding, proportional)
            umesh.update_tag = bool(islands)

    @staticmethod
    def crop_inplace(axis, padding, umeshes, proportional, inplace=True, extended=True):
        islands_of_tile: dict[int, list[tuple[FaceIsland, BBox]]] = {}
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh, extended=extended):
                for island in islands:
                    bbox = island.calc_bbox()
                    islands_of_tile.setdefault(bbox.tile_from_center, []).append((island, bbox))
            umesh.update_tag = bool(islands)

        if not islands_of_tile:
            return

        for tile, islands_and_bboxes in islands_of_tile.items():
            islands = []
            general_bbox = BBox()
            for island, bbox in islands_and_bboxes:
                islands.append(island)
                general_bbox.union(bbox)

            UNIV_OT_Crop.crop_ex(axis, general_bbox, inplace, islands, Vector((0, 0)), padding, proportional)

    @staticmethod
    def crop_ex(axis, bbox, inplace, islands_of_mesh, offset, padding, proportional):
        scale_x = ((1.0 - padding) / w) if (w := bbox.width) else 1
        scale_y = ((1.0 - padding) / h) if (h := bbox.height) else 1

        if proportional:
            if axis == 'XY':
                scale_x = scale_y = min(scale_x, scale_y)
            elif axis == 'X':
                scale_x = scale_y = scale_x
            else:
                scale_x = scale_y = scale_y
        else:
            if axis == 'X':
                scale_y = 1
            if axis == 'Y':
                scale_x = 1

        scale = Vector((scale_x, scale_y))
        bbox.scale(scale)

        pos_x = utils.wrap_line(bbox.min.x, bbox.width+padding, 0, 1, default=0)
        pos_y = utils.wrap_line(bbox.min.y, bbox.height+padding, 0, 1, default=0)

        set_pos = Vector((pos_x, pos_y)) + Vector((padding, padding)) / 2
        set_pos += offset
        if axis == 'XY':
            if inplace:
                set_pos += bbox.tile_from_center
        elif axis == 'X':
            set_pos.y = 0
            if inplace:
                set_pos.x += math.floor(bbox.center_x)
        else:
            set_pos.x = 0
            if inplace:
                set_pos.y += math.floor(bbox.center_y)

        for islands in islands_of_mesh:
            islands.scale(scale, bbox.center)
            islands.set_position(set_pos, bbox.min)

    @staticmethod
    def get_event_info():
        return info.operator.crop_event_info_ex


class UNIV_OT_Fill(UNIV_OT_Crop):
    bl_idname = 'uv.univ_fill'
    bl_label = 'Fill'
    bl_description = info.operator.fill_info
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        self.mode_preprocessing()
        self.calc_padding()
        self.report_padding()
        return self.crop(self.mode, self.axis, self.padding, proportional=False, report=self.report)

    @staticmethod
    def get_event_info():
        return info.operator.fill_event_info_ex

class Align_by_Angle:

    angle: FloatProperty(name='Angle', default=to_rad(15), min=to_rad(2), max=to_rad(40), soft_min=to_rad(5), subtype='ANGLE')

    def align_edge_by_angle(self, x_axis):
        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()  # noqa
        umeshes: types.UMeshes = selected_umeshes if selected_umeshes else visible_umeshes
        umeshes.fix_context()
        umeshes.calc_aspect_ratio(from_mesh=False)

        if not umeshes:
            return umeshes.update()
        umeshes.update_tag = False

        has_segments = False
        for umesh in umeshes:
            uv = umesh.uv
            edge_orient = Vector((not x_axis, x_axis))

            angle = self.angle
            negative_ange = math.pi - angle

            groups = []
            islands = AdvIslands.calc_visible(umesh)
            islands.indexing()

            for isl in islands:
                isl.apply_aspect_ratio()
                if selected_umeshes:
                    if umeshes.sync:
                        if umeshes.elem_mode == 'FACE' or umesh.total_face_sel:
                            def corners_iter():
                                for crn_ in isl.corners_iter():
                                    if crn_.edge.select:
                                        if crn_.face.select or (utils.is_pair(crn_, crn_.link_loop_prev, uv) and crn_.link_loop_prev.face.select):
                                            yield crn_
                                        else:
                                            crn_.tag = False
                                    else:
                                        crn_.tag = False
                        else:
                            def corners_iter():
                                for crn_ in isl.corners_iter():
                                    if crn_.edge.select:
                                        yield crn_
                                    else:
                                        crn_.tag = False
                    else:
                        def corners_iter():
                            for crn_ in isl.corners_iter():
                                if not crn_[uv].select_edge:
                                    crn_.tag = False
                                    continue
                                yield crn_
                else:
                    def corners_iter():
                        return isl.corners_iter()

                to_segmenting_corners = []
                for crn in corners_iter():
                    vec = crn[uv].uv - crn.link_loop_next[uv].uv
                    a = vec.angle(edge_orient, 0)

                    if a <= angle or a >= negative_ange:
                        to_segmenting_corners.append(crn)
                        crn.tag = True
                    else:
                        crn.tag = False

                    groups.append(to_segmenting_corners)

                has_segments |= bool(to_segmenting_corners)
                segments = types.Segments.from_tagged_corners(to_segmenting_corners, umesh)
                segments = segments.break_by_cardinal_dir()
                segments.segments.sort(key=lambda seg__: seg__.length)
                segments.segments.sort(key=lambda seg__: seg__.weight_angle, reverse=True)

                new_segments = self.join_segments_by_angle(segments)
                self.align_by_angle_ex(new_segments, x_axis)
                isl.reset_aspect_ratio()

        if not has_segments:
            self.report({'INFO'}, f'Not found edges with {math.degrees(self.angle):.1f} angle')  # noqa
        elif not umeshes.update_tag:
            self.report({'INFO'}, 'All edges aligned')  # noqa
        umeshes.silent_update()
        return {'FINISHED'}

    @staticmethod
    def align_by_angle_ex(segments: types.Segments, x_axis=True):
        uv = segments.umesh.uv
        for idx, seg in enumerate(segments):
            if seg.is_end_lock:
                seg.lengths_seq.pop()

            if not seg.lengths_seq:
                continue

            if seg.is_start_lock:
                del seg.lengths_seq[0]
            if not seg.lengths_seq:
                continue

            seg.calc_chain_linked_corners()
            center_coords = []
            it = iter(seg.chain_linked_corners)

            if x_axis:
                prev_co = next(it)[0][uv].uv.x
                for chain in it:
                    curr_co = chain[0][uv].uv.x
                    center_coords.append((prev_co + curr_co) * 0.5)
                    prev_co = curr_co
            else:
                prev_co = next(it)[0][uv].uv.y
                for chain in it:
                    curr_co = chain[0][uv].uv.y
                    center_coords.append((prev_co + curr_co) * 0.5)
                    prev_co = curr_co

            try:
                component: float = np.average(center_coords, weights=seg.lengths_seq)  # noqa
            except ZeroDivisionError:
                continue

            all_equal = True
            if x_axis:
                for chain in seg.chain_linked_corners:
                    if not isclose(chain[0][uv].uv.x, component, abs_tol=1e-6):
                        all_equal = False
                        for crn in chain:
                            crn[uv].uv.x = component
            else:
                for chain in seg.chain_linked_corners:
                    if not isclose(chain[0][uv].uv.y, component, abs_tol=1e-6):
                        all_equal = False
                        for crn in chain:
                            crn[uv].uv.y = component
            if not all_equal:
                segments.umesh.update_tag = True

    def join_segments_by_angle(self, segments: types.Segments):  # noqa
        new_segments = []
        while segments:
            tar_seg: types.Segment = segments.segments.pop()
            if not tar_seg.tag:  # Skip joined segments
                continue
            tar_seg.tag = False

            if tar_seg.is_start_lock and tar_seg.is_end_lock:
                new_segments.append(tar_seg)
                continue

            tar_start_vert = tar_seg.start_vert
            tar_end_vert = tar_seg.end_vert

            tar_start_co = tar_seg.start_co
            tar_end_co = tar_seg.end_co

            grow_from_start = []
            grow_from_end = []

            # Collect and reverse segments that can be joined together
            for seg in reversed(segments):
                if not seg.tag:
                    continue

                end_vert = seg.end_vert
                end_co = seg.end_co

                # Grow from start
                if not tar_seg.is_start_lock:
                    if tar_start_vert == end_vert:
                        if tar_start_co == end_co:
                            grow_from_start.append(seg)
                            continue
                    elif tar_start_vert == seg.start_vert:
                        if tar_start_co == seg.start_co:
                            seg.reverse()
                            grow_from_start.append(seg)
                            continue

                # Grow from end
                if not tar_seg.is_end_lock:
                    if tar_end_vert == seg.start_vert:
                        if tar_end_co == seg.start_co:
                            grow_from_end.append(seg)
                            continue
                    elif tar_end_vert == end_vert:
                        if tar_end_co == end_co:
                            seg.reverse()
                            grow_from_end.append(seg)
                            continue

            Align_by_Angle.join_segments_by_optimal_angle(grow_from_end, grow_from_start, new_segments, segments, tar_seg)

        return types.Segments(new_segments, segments.umesh)

    @staticmethod
    def join_segments_by_optimal_angle(grow_from_end, grow_from_start, new_segments, segments, tar_seg):
        """Connecting the segments at the optimal angle"""
        is_joined = False
        # Save the start and end variables in advance, as the segment may be joined and
        # the result will be wrong for cutting off non-priority segments.
        tar_seg_start = tar_seg.start
        tar_seg_end = tar_seg.end

        if grow_from_end:
            tar_vec = tar_seg[-1].vec
            card_vec = utils.vec_to_cardinal(tar_vec)

            for seg in reversed(grow_from_end):
                if card_vec != utils.vec_to_cardinal(seg[0].vec):
                    seg.is_start_lock = True
                    # Check the other end, since segments can be cyclic, in which case you need to lock.
                    seg.is_end_lock |= tar_seg_start == seg.end
                    grow_from_end.remove(seg)

            if not grow_from_end:
                tar_seg.is_end_lock = True

            for seg in grow_from_end:
                seg.value = card_vec.angle_signed(seg[0].vec)
            Align_by_Angle.preserving_identical_oppositely_angles(grow_from_end, start_lock=True)

            min_seg = None
            min_angle = float('inf')

            for seg in grow_from_end:
                if seg.is_start_lock:
                    continue
                if (min_a := tar_vec.angle(seg[0].vec)) < min_angle:
                    min_angle = min_a
                    min_seg = seg

            # Lock not joined segments
            if min_seg is not None:
                for seg in grow_from_end:
                    if not (seg is min_seg):
                        seg.is_start_lock = True
                        # Check the other end, since segments can be cyclic, in which case you need to lock.
                        seg.is_end_lock |= tar_seg_start == seg.end

                tar_seg.join_from_end(min_seg)
                tar_seg.tag = True
                is_joined = True

        assert not tar_seg.is_circular

        if grow_from_start:
            tar_vec = tar_seg[0].vec
            card_vec = utils.vec_to_cardinal(tar_vec)

            for seg in reversed(grow_from_start):
                if card_vec != utils.vec_to_cardinal(seg[-1].vec):
                    seg.is_end_lock = True
                    # Check the other end, since segments can be cyclic, in which case you need to lock.
                    seg.is_start_lock |= tar_seg_end == seg.start
                    grow_from_start.remove(seg)

            if not grow_from_start:
                tar_seg.is_start_lock = True

            for seg in grow_from_start:
                seg.value = card_vec.angle_signed(seg[-1].vec, 0)

            Align_by_Angle.preserving_identical_oppositely_angles(grow_from_start, start_lock=False)

            min_seg = None
            min_angle = float('inf')

            for seg in grow_from_start:
                if seg.is_end_lock:
                    continue
                if (min_a := tar_vec.angle(seg[-1].vec, 0)) < min_angle:
                    min_angle = min_a
                    min_seg = seg

            if min_seg is not None:
                for seg in grow_from_start:
                    if not (seg is min_seg):
                        seg.is_end_lock = True
                        # Check the other end, since segments can be cyclic, in which case you need to lock.
                        seg.is_start_lock |= tar_seg_end == seg.start

                tar_seg.tag = False
                min_seg.join_from_end(tar_seg)
                min_seg.tag = True
                tar_seg = min_seg
                is_joined = True

        if is_joined:
            segments.segments.append(tar_seg)
        else:
            new_segments.append(tar_seg)

    @staticmethod
    def preserving_identical_oppositely_angles(grow_from, start_lock):
        # Preserving segments with identical but oppositely directed angles
        for seg_a in grow_from:
            for seg_b in grow_from:
                if seg_a is seg_b or np.sign(seg_a.value) == np.sign(seg_b.value):
                    continue

                if isclose(seg_a.value, -seg_b.value, abs_tol=to_rad(1.5)):
                    if start_lock:
                        seg_a.is_start_lock = True
                        seg_b.is_start_lock = True
                    else:
                        seg_a.is_end_lock = True
                        seg_b.is_end_lock = True


class Collect(utils.OverlapHelper):
    def collect_islands(self):
        settings = univ_settings()
        padding = settings.padding / min(int(settings.size_x), int(settings.size_y))
        padding = bl_math.clamp(padding, 0.001, float('inf'))
        padding *= 2

        umeshes = UMeshes.calc()
        umeshes.fix_context()

        selected_umeshes, visible_umeshes = umeshes.filtered_by_selected_and_visible_uv_faces()
        umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if selected_umeshes:
            islands_calc_type = AdvIslands.calc_selected_with_mark_seam
        else:
            islands_calc_type = AdvIslands.calc_visible_with_mark_seam

        all_islands = []
        for umesh in umeshes:
            if adv_islands := islands_calc_type(umesh):
                if self.lock_overlap_mode == 'ANY':
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_uv_coords(save_triplet=True)
                all_islands.extend(adv_islands)
            umesh.update_tag = bool(adv_islands)

        if not all_islands:
            self.report({'WARNING'}, 'Islands not found')  # noqa
            return {'FINISHED'}

        if self.lock_overlap:
            threshold = self.threshold if self.lock_overlap_mode == 'EXACT' else None
            all_islands = UnionIslands.calc_overlapped_island_groups(all_islands, threshold)

        general_bbox = BBox()
        for isl in all_islands:
            general_bbox.union(isl.bbox)

        # Sort by center
        general_center = general_bbox.center
        all_islands.sort(key = lambda isl_: (isl_.bbox.center - general_center).length)

        # Scale to avoid small padding
        all_bbox = []
        for isl in all_islands:
            bb = isl.bbox
            bb.pad(Vector((padding, padding)) * 0.52)
            all_bbox.append(bb)

        radius = self.estimate_min_radius(all_bbox, 1.2 + padding)
        boxes = self.pack_bboxes(all_bbox, step=padding, max_radius = radius)

        placed = []
        failed_to_place = []
        for isl, new_bb, old_bb in zip(all_islands, boxes, all_bbox):
            if new_bb is None:
                failed_to_place.append(isl)
            else:
                placed.append(isl)
                to = new_bb.center
                _from = old_bb.center
                delta = (to - _from) + general_center
                isl.move(delta)

        if failed_to_place:
            if not placed:
                self.report({'WARNING'}, "Failed inplace packing")  # noqa
                return {'FINISHED'}
            else:
                self.report({'INFO'}, "Some islands couldn't be packed and were placed to the right.")  # noqa

            placed_bbox = BBox()
            for isl in placed:
                placed_bbox.union(isl.calc_bbox())
            placed_bbox.pad(Vector((padding, padding)))

            right_center = (placed_bbox.right_upper + placed_bbox.right_bottom) / 2
            for failed_isl in failed_to_place:
                right_center.x += padding

                bb = failed_isl.calc_bbox()
                left_center = (bb.left_upper + bb.left_bottom) / 2

                failed_isl.set_position(right_center, left_center)
                right_center.x += bb.width

        return umeshes.update()

    @staticmethod
    def generate_search_positions(max_radius: float, step: float):
        r = int(max_radius / step)
        if r > 150:
            new_r = 150
            step = max_radius / new_r
            r = new_r


        range_xy = np.arange(-r, r + 1)  # dtype='int16'
        grid_x, grid_y = np.meshgrid(range_xy, range_xy)

        dist = np.hypot(grid_x, grid_y)

        coords = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
        distances = dist.flatten()

        coords = coords[np.argsort(distances)].astype('float32')
        coords *= np.float32(step)  # dtype='float32'
        return coords

    def pack_bboxes(self, rects: list[BBox], step, max_radius) -> list[BBox | None]:
        placed = []
        failed_idx = []
        positions = self.generate_search_positions(max_radius, step)

        for bbox in rects:
            for pos in positions:
                candidate = bbox.moved(Vector(pos))
                for r in reversed(placed):
                    if candidate.overlap(r):
                        break
                else:
                    placed.append(candidate)
                    positions = self.subtract_points_by_bbox(positions, candidate)
                    break
            else:
                failed_idx.append(len(placed))

        for idx in reversed(failed_idx):
            placed.insert(idx, None)

        return placed

    @staticmethod
    def subtract_points_by_bbox(coords, bbox):
        bb_min = np.array(bbox.min, dtype='float32')
        bb_max = np.array(bbox.max, dtype='float32')
        mask = np.all((coords >= bb_min) & (coords <= bb_max), axis=1)
        return coords[~mask]

    @staticmethod
    def estimate_min_radius(bboxes, margin=1.2):
        total_area = sum(bb.area for bb in bboxes)
        max_length = max(bb.max_length for bb in bboxes) / 2 * 1.2
        radius = math.sqrt(total_area / math.pi)
        max_radius = max(radius, max_length)
        return max_radius * margin

align_align_direction_items = (
    ('UPPER', 'Upper', ''),
    ('BOTTOM', 'Bottom', ''),
    ('LEFT', 'Left', ''),
    ('RIGHT', 'Right', ''),
    ('CENTER', 'Center', ''),
    ('HORIZONTAL', 'Horizontal', ''),
    ('VERTICAL', 'Vertical', ''),
    ('LEFT_UPPER', 'Left upper', ''),
    ('RIGHT_UPPER', 'Right upper', ''),
    ('LEFT_BOTTOM', 'Left bottom', ''),
    ('RIGHT_BOTTOM', 'Right bottom', ''),
)
class UNIV_OT_Align_pie(Operator, Collect, Align_by_Angle):
    bl_idname = 'uv.univ_align_pie'
    bl_label = 'Align'
    bl_description = "Align verts, edges, faces, islands and cursor"
    bl_options = {'REGISTER', 'UNDO'}

    direction: EnumProperty(name="Direction", default='UPPER', items=align_align_direction_items)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        if self.is_island_mode:
            if self.mode == 'INDIVIDUAL_OR_MOVE':
                if self.direction == 'CENTER':
                    self.layout.label(text='Collect')
                    self.draw_overlap()
                    self.layout.separator()
                elif self.direction in ('HORIZONTAL', 'VERTICAL'):
                    self.layout.label(text='Align Edge by Angle')
                    self.layout.prop(self, 'angle', slider=True)
                    self.layout.separator()

        self.layout.prop(self, 'direction')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes = None
        self.is_island_mode = False

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
        settings = univ_settings()
        self.mode = settings.align_mode  # noqa
        if settings.align_island_mode == 'FOLLOW':
            self.is_island_mode = utils.is_island_mode()
        else:
            self.is_island_mode = settings.align_island_mode == 'ISLAND'
        return self.align()

    def align(self):
        match self.mode:
            case 'ALIGN':
                self.align_ex(selected=True)
                if not self.umeshes.final():
                    self.align_ex(selected=False)

            case 'ALIGN_TO_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    self.umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                self.move_to_cursor_ex(cursor_loc, selected=True)
                if not self.umeshes.final():
                    self.move_to_cursor_ex(cursor_loc, selected=False)

            case 'ALIGN_TO_CURSOR_UNION':
                if not (cursor_loc := utils.get_cursor_location()):
                    self.umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                self.move_to_cursor_union_ex(cursor_loc, selected=True)
                if not self.umeshes.final():
                    self.move_to_cursor_union_ex(cursor_loc, selected=False)

            case 'ALIGN_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    self.umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                general_bbox = self.align_cursor_ex(selected=True)
                if not general_bbox.is_valid:
                    general_bbox = self.align_cursor_ex(selected=False)
                if not general_bbox.is_valid:
                    self.umeshes.report()
                    return {'CANCELLED'}
                self.align_cursor(general_bbox, cursor_loc)
                return {'FINISHED'}

            case 'INDIVIDUAL_OR_MOVE':
                if self.is_island_mode:
                    if self.direction == 'CENTER':
                        return self.collect_islands()
                    elif self.direction in ('HORIZONTAL', 'VERTICAL'):
                        return self.align_edge_by_angle(x_axis=self.direction == 'VERTICAL')
                    else:
                        self.move_ex(selected=True)
                else:
                    self.individual_scale_zero()

                if not self.umeshes.final():
                    self.move_ex(selected=False)

            case _:
                raise NotImplementedError(self.mode)

        return self.umeshes.update()

    def move_to_cursor_ex(self, cursor_loc, selected=True):
        all_groups = []  # islands, bboxes, uv or corners, uv
        general_bbox = BBox.init_from_minmax(cursor_loc, cursor_loc)
        if self.is_island_mode or (not selected and self.direction not in {'LEFT', 'RIGHT', 'BOTTOM', 'UPPER'}):
            for umesh in self.umeshes:
                if islands := Islands.calc(umesh, selected=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        all_groups.append((island, bbox, umesh.uv))
                umesh.update_tag = bool(islands)
            self.align_islands(all_groups, general_bbox, invert=True)
        else:
            for umesh in self.umeshes:
                if corners := utils.calc_uv_corners(umesh, selected=selected):
                    all_groups.append((corners, umesh.uv))
                umesh.update_tag = bool(corners)
            self.align_corners(all_groups, general_bbox)

    def move_to_cursor_union_ex(self, cursor_loc, selected=True):
        all_groups = []  # islands, bboxes, uv or corners, uv
        target_bbox = BBox.init_from_minmax(cursor_loc, cursor_loc)
        general_bbox = BBox()
        for umesh in self.umeshes:
            if faces := utils.calc_uv_faces(umesh, selected=selected):
                island = FaceIsland(faces, umesh)
                bbox = island.calc_bbox()
                general_bbox.union(bbox)
                all_groups.append([island, bbox, umesh.uv])
            umesh.update_tag = bool(faces)
        for group in all_groups:
            group[1] = general_bbox
        self.align_islands(all_groups, target_bbox, invert=True)

    def align_cursor_ex(self, selected):
        general_bbox = BBox()
        if self.umeshes.sync and selected and any(umesh.total_edge_sel for umesh in self.umeshes):
            for umesh in self.umeshes:
                if umesh.elem_mode in ('FACE', 'ISLAND'):
                    corners = [crn for f in utils.calc_selected_uv_faces_iter(umesh) for crn in f.loops]
                else:
                    corners = utils.calc_selected_uv_edge_corners(umesh)
                if corners:
                    uv = umesh.uv
                    general_bbox.update(crn[uv].uv for crn in corners)
                    general_bbox.update(crn.link_loop_next[uv].uv for crn in corners)
        else:
            for umesh in self.umeshes:
                if corners := utils.calc_uv_corners(umesh, selected=selected):
                    uv = umesh.uv
                    general_bbox.update(crn[uv].uv for crn in corners)
        return general_bbox

    @staticmethod
    def get_unique_linked_corners_from_crn_edge(umesh, corners):
        assert umesh.sync
        uv = umesh.uv
        unique_linked_corners = set()
        for crn in corners:
            if crn not in unique_linked_corners:
                unique_linked_corners.add(crn)
                unique_linked_corners.update(utils.linked_crn_to_vert_pair_with_seam(crn, uv, True))
            next_crn = crn.link_loop_next
            if next_crn not in unique_linked_corners:
                unique_linked_corners.add(next_crn)
                unique_linked_corners.update(utils.linked_crn_to_vert_pair_with_seam(next_crn, uv, True))
        return unique_linked_corners

    def align_ex(self, selected=True):
        all_groups = []  # islands, bboxes, uv or corners, uv
        general_bbox = BBox()
        if self.is_island_mode or not selected:
            for umesh in self.umeshes:
                if islands := Islands.calc_extended_or_visible_with_mark_seam(umesh, extended=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        general_bbox.union(bbox)

                        all_groups.append((island, bbox, umesh.uv))
                umesh.update_tag = bool(islands)
            self.align_islands(all_groups, general_bbox)
        else:
            for umesh in self.umeshes:
                if umesh.sync and any(umesh.total_edge_sel for umesh in self.umeshes):
                    if umesh.elem_mode in ('FACE', 'ISLAND'):
                        corners = [crn for f in utils.calc_selected_uv_faces_iter(umesh) for crn in f.loops]
                    else:
                        corners = utils.calc_selected_uv_edge_corners(umesh)
                    if corners:
                        corners = self.get_unique_linked_corners_from_crn_edge(umesh, corners)
                        bbox = BBox.calc_bbox_uv_corners(corners, umesh.uv)
                        general_bbox.union(bbox)
                        all_groups.append((corners, umesh.uv))
                else:
                    if corners := utils.calc_selected_uv_vert_corners(umesh):
                        bbox = BBox.calc_bbox_uv_corners(corners, umesh.uv)
                        general_bbox.union(bbox)
                        all_groups.append((corners, umesh.uv))
                    umesh.update_tag = bool(corners)
            self.align_corners(all_groups, general_bbox)

    def move_ex(self, selected=True):
        assert self.direction not in {'CENTER', 'HORIZONTAL', 'VERTICAL'}
        move_value = Vector(self.get_move_value(self.direction))
        if self.is_island_mode:
            for umesh in self.umeshes:
                if islands := Islands.calc_extended_or_visible(umesh, extended=selected):
                    for island in islands:
                        island.move(move_value)
                umesh.update_tag = bool(islands)
        else:
            for umesh in self.umeshes:
                if corners := utils.calc_uv_corners(umesh, selected=selected):
                    uv = umesh.uv
                    for corner in corners:
                        corner[uv].uv += move_value
                umesh.update_tag = bool(corners)

    def individual_scale_zero(self):
        if self.umeshes.elem_mode == 'FACE':
            for umesh in self.umeshes:
                uv = umesh.uv
                if islands := Islands.calc_selected_with_mark_seam(umesh):
                    for isl in islands:
                        self.align_corners(((isl.corners_iter(), uv),), isl.calc_bbox())
                umesh.update_tag = bool(islands)
        else:
            for umesh in self.umeshes:
                uv = umesh.uv
                if lgs := types.LoopGroup.calc_dirt_loop_groups(umesh):
                    umesh.tag_visible_corners()
                    for lg in lgs:
                        lg.extend_from_linked()
                        self.align_corners(((lg, uv),), lg.calc_bbox())
                umesh.update_tag = bool(lgs)

    def align_islands(self, groups, general_bbox, invert=False):
        for island, bounds, _ in groups:
            center = bounds.center
            match self.direction:
                case 'UPPER':
                    delta = (0, (general_bbox.min - bounds.min).y) if invert else (0, (general_bbox.max - bounds.max).y)
                case 'BOTTOM':
                    delta = (0, (general_bbox.max - bounds.max).y) if invert else (0, (general_bbox.min - bounds.min).y)
                case 'LEFT':
                    delta = ((general_bbox.max - bounds.max).x, 0) if invert else ((general_bbox.min - bounds.min).x, 0)
                case 'RIGHT':
                    delta = ((general_bbox.min - bounds.min).x, 0) if invert else ((general_bbox.max - bounds.max).x, 0)
                case 'CENTER':
                    delta = general_bbox.center - center
                case 'HORIZONTAL':
                    delta = (0, (general_bbox.center - center).y)
                case 'VERTICAL':
                    delta = (general_bbox.center - center).x, 0
                case 'RIGHT_UPPER':
                    delta = general_bbox.min - bounds.min if invert else general_bbox.max - bounds.max
                case 'LEFT_UPPER':
                    if invert:
                        delta = (general_bbox.max - bounds.max).x, (general_bbox.min - bounds.min).y
                    else:
                        delta = (general_bbox.min - bounds.min).x, (general_bbox.max - bounds.max).y
                case 'LEFT_BOTTOM':
                    delta = general_bbox.max - bounds.max if invert else general_bbox.min - bounds.min
                case 'RIGHT_BOTTOM':
                    if invert:
                        delta = (general_bbox.min - bounds.min).x, (general_bbox.max - bounds.max).y
                    else:
                        delta = (general_bbox.max - bounds.max).x, (general_bbox.min - bounds.min).y
                case _:
                    raise NotImplementedError(self.direction)
            island.move(Vector(delta))

    def align_corners(self, groups, general_bbox):
        match self.direction:
            case 'LEFT' | 'RIGHT' | 'VERTICAL':
                if self.direction == 'LEFT':
                    destination = general_bbox.min.x
                elif self.direction == 'RIGHT':
                    destination = general_bbox.max.x
                else:
                    destination = general_bbox.center.x

                for luvs, uv in groups:
                    for luv in luvs:
                        luv[uv].uv.x = destination
            case 'UPPER' | 'BOTTOM' | 'HORIZONTAL':
                if self.direction == 'UPPER':
                    destination = general_bbox.max.y
                elif self.direction == 'BOTTOM':
                    destination = general_bbox.min.y
                else:
                    destination = general_bbox.center.y

                for luvs, uv in groups:
                    for luv in luvs:
                        luv[uv].uv[1] = destination
            case _:
                if self.direction == 'CENTER':
                    destination = general_bbox.center
                elif self.direction == 'LEFT_BOTTOM':
                    destination = general_bbox.left_bottom
                elif self.direction == 'RIGHT_UPPER':
                    destination = general_bbox.right_upper
                elif self.direction == 'LEFT_UPPER':
                    destination = general_bbox.left_upper
                elif self.direction == 'RIGHT_BOTTOM':
                    destination = general_bbox.right_bottom
                else:
                    raise NotImplementedError(self.direction)

                for luvs, uv in groups:
                    for luv in luvs:
                        luv[uv].uv = destination

    def align_cursor(self, general_bbox, cursor_loc):
        if self.direction in ('UPPER', 'BOTTOM'):
            loc = getattr(general_bbox, self.direction.lower())
            loc.x = cursor_loc.x
            utils.set_cursor_location(loc)
        elif self.direction in ('RIGHT', 'LEFT'):
            loc = getattr(general_bbox, self.direction.lower())
            loc.y = cursor_loc.y
            utils.set_cursor_location(loc)
        elif loc := getattr(general_bbox, self.direction.lower(), False):
            utils.set_cursor_location(loc)
        elif self.direction == 'VERTICAL':
            utils.set_cursor_location(Vector((general_bbox.center.x, cursor_loc.y)))
        elif self.direction == 'HORIZONTAL':
            utils.set_cursor_location(Vector((cursor_loc.x, general_bbox.center.y)))
        else:
            raise NotImplementedError(self.direction)

    @staticmethod
    def get_move_value(direction):
        match direction:
            case 'UPPER':
                return 0, 1
            case 'BOTTOM':
                return 0, -1
            case 'LEFT':
                return -1, 0
            case 'RIGHT':
                return 1, 0
            case 'RIGHT_UPPER':
                return 1, 1
            case 'LEFT_UPPER':
                return -1, 1
            case 'LEFT_BOTTOM':
                return -1, -1
            case 'RIGHT_BOTTOM':
                return 1, -1
            case _:
                raise NotImplementedError(direction)

align_event_info_ex = \
        "Default - Align faces/verts\n" \
        "Shift - Arrows and H/V align vertices or edges individually in Vertex/Edge mode.\n" \
         "\t\t\tIn Island mode, they move entire islands.\n" \
         "\t\t\tCenter button collects islands in Island mode.\n" \
         "\t\t\tH/V buttons - align edges by angle in Island mode.\n" \
        "Ctrl - Align to cursor\n" \
        "Ctrl+Shift+Alt - Align to cursor union\n" \
        "Alt - Move cursor to selected faces/verts"
        # "Ctrl+Shift+LMB = Collision move (Not Implement)\n"

class UNIV_OT_Align(UNIV_OT_Align_pie):
    bl_idname = 'uv.univ_align'
    bl_description = "Align verts, edges, faces, islands and cursor \n\n" + align_event_info_ex

    mode: EnumProperty(name="Mode", default='ALIGN', items=(
        ('ALIGN', 'Align', ''),
        ('INDIVIDUAL_OR_MOVE', 'Individual | Move', ''),
        ('ALIGN_CURSOR', 'Move cursor to selected', ''),
        ('ALIGN_TO_CURSOR', 'Align to cursor', ''),
        ('ALIGN_TO_CURSOR_UNION', 'Align to cursor union', ''),
        # ('MOVE_COLLISION', 'Collision move', '')
    ))

    def draw(self, context):
        if self.is_island_mode:
            if self.mode == 'INDIVIDUAL_OR_MOVE':
                if self.direction == 'CENTER':
                    self.layout.label(text='Collect')
                    self.draw_overlap()
                    self.layout.separator()
                elif self.direction in ('HORIZONTAL', 'VERTICAL'):
                    self.layout.label(text='Align by Angle')
                    self.layout.prop(self, 'angle', slider=True)
                    self.layout.separator()

        self.layout.prop(self, 'direction')
        self.layout.column(align=True).prop(self, 'mode', expand=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        match event.ctrl, event.shift, event.alt:
            case False, False, False:
                self.mode = 'ALIGN'
            case True, False, False:
                self.mode = 'ALIGN_TO_CURSOR'
            case True, True, True:
                self.mode = 'ALIGN_TO_CURSOR_UNION'
            case False, False, True:
                self.mode = 'ALIGN_CURSOR'
            case False, True, False:
                self.mode = 'INDIVIDUAL_OR_MOVE'
            case _:
                self.report({'INFO'}, f"Event: {utils.event_to_string(event)} not implement. \n\n"
                                      f"See all variations:\n\n{align_event_info_ex}")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
        self.is_island_mode = utils.is_island_mode()
        return self.align()


class UNIV_OT_Flip(Operator):
    bl_idname = 'uv.univ_flip'
    bl_label = 'Flip'
    bl_description = "FlipX and FlipY.\n\n" \
                     "Default - Flip island.\n" \
                     "Shift - Individual flip.\n" \
                     "Ctrl - Flip by cursor.\n" \
                     "Alt - Flip by Y axis.\n\n" \
                     "Shift and Ctrl conflict between them"

    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='DEFAULT', items=(
        ('DEFAULT', 'Default', ''),
        ('BY_CURSOR', 'By cursor', ''),
        ('INDIVIDUAL', 'Individual', ''),
        ('FLIPPED', 'Flipped', ''),
    ))

    axis: EnumProperty(name='Axis', default='X', items=(('X', 'X', ''), ('Y', 'Y', '')))

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.row(align=True).prop(self, 'axis', expand=True)
        self.layout.column(align=True).prop(self, 'mode', expand=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            if context.area.type == 'IMAGE_EDITOR' and context.area.ui_type == 'UV':
                self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
                self.mouse_pos = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)
        self.axis = 'Y' if event.alt else 'X'
        match event.ctrl, event.shift:
            case False, False:
                self.mode = 'DEFAULT'
            case True, False:
                self.mode = 'BY_CURSOR'
            case False, True:
                self.mode = 'INDIVIDUAL'
            case _:
                self.report({'INFO'}, f"Event: {utils.event_to_string(event)} not implement. \n\n")
                return {'CANCELLED'}
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None
        self.scale = Vector((1, 1))
        self.max_distance: float = 0.0
        self.mouse_pos: Vector | None = None

    def execute(self, context):
        self.umeshes = UMeshes(report=self.report)
        self.scale = self.get_flip_scale_from_axis(self.axis)

        self.umeshes = UMeshes(report=self.report)
        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if not self.umeshes:
            return self.umeshes.update()
        if not selected_umeshes and self.mouse_pos:
            return self.pick_flip()

        match self.mode:
            case 'DEFAULT':
                self.flip_ex(extended=selected_umeshes)
            case 'BY_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    self.umeshes.report({'INFO'}, "Cursor not found")
                    return {'FINISHED'}
                self.flip_by_cursor(cursor=cursor_loc, extended=selected_umeshes)
            case 'INDIVIDUAL':
                self.flip_individual(extended=selected_umeshes)
            case _: # 'FLIPPED':
                self.flip_flipped(extended=selected_umeshes)
        return self.umeshes.update()

    def flip_ex(self, extended):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in self.umeshes:
            if islands := Islands.calc_extended_or_visible_with_mark_seam(umesh, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
            umesh.update_tag = bool(islands)

        pivot = general_bbox.center
        for islands in islands_of_mesh:
            islands.scale(scale=self.scale, pivot=pivot)

    def flip_by_cursor(self, cursor, extended):
        for umesh in self.umeshes:
            if islands := Islands.calc_extended_or_visible_with_mark_seam(umesh, extended=extended):
                islands.scale(scale=self.scale, pivot=cursor)
            umesh.update_tag = bool(islands)

    def flip_individual(self, extended):
        for umesh in self.umeshes:
            if islands := Islands.calc_extended_or_visible_with_mark_seam(umesh, extended=extended):
                for island in islands:
                    island.scale(scale=self.scale, pivot=island.calc_bbox().center)
            umesh.update_tag = bool(islands)

    def flip_flipped(self, extended):
        for umesh in self.umeshes:
            if islands := self.calc_extended_or_visible_flipped_islands_with_mark_seam(umesh, extended=extended):
                for island in islands:
                    island.scale(scale=self.scale, pivot=island.calc_bbox().center)
            umesh.update_tag = bool(islands)

        if not self.umeshes.update_tag:
            return self.report({'INFO'}, 'Flipped islands not found')

    @staticmethod
    def calc_extended_or_visible_flipped_islands_with_mark_seam(umesh: types.UMesh, extended):
        uv = umesh.uv
        if extended:
            if umesh.is_full_face_deselected:
                return AdvIslands()

        AdvIslands.tag_filter_visible(umesh)

        for f_ in umesh.bm.faces:
            if f_.tag:
                f_.tag = utils.is_flipped_uv(f_, uv)

        if extended:
            islands_ = [AdvIslands.island_type(i, umesh) for i in AdvIslands.calc_with_markseam_iter_ex(umesh) if
                       AdvIslands.island_filter_is_any_face_selected(i, umesh)]
        else:
            islands_ = [AdvIslands.island_type(i, umesh) for i in AdvIslands.calc_with_markseam_iter_ex(umesh)]
        return AdvIslands(islands_, umesh)

    @staticmethod
    def get_flip_scale_from_axis(axis):
        return Vector((-1, 1)) if axis == 'X' else Vector((1, -1))

    def pick_flip(self):
        hit = types.IslandHit(self.mouse_pos, self.max_distance)
        if self.mode == 'FLIPPED':
            islands_calc_type = self.calc_extended_or_visible_flipped_islands_with_mark_seam
        else:
            islands_calc_type = AdvIslands.calc_extended_or_visible_with_mark_seam

        for umesh in self.umeshes:
            for isl in islands_calc_type(umesh, extended=False):
                hit.find_nearest_island_by_crn(isl)

        if not hit:
            message = 'Island not found within a given radius'
            if self.mode == 'FLIPPED':
                self.report({'INFO'}, 'Flipped ' + message.lower())
            else:
                self.report({'INFO'}, message)
            return {'CANCELLED'}

        pivot = utils.get_cursor_location() if self.mode == 'BY_CURSOR' else hit.island.bbox.center
        hit.island.scale(scale=self.scale, pivot=pivot)
        hit.island.umesh.update()
        return {'FINISHED'}


class UNIV_OT_Rotate(Operator):
    bl_idname = 'uv.univ_rotate'
    bl_label = 'Rotate'
    bl_description = "Rotate CW and Rotate CCW\n\n" \
                     "Context keymaps on button:\n" \
                     "\t\tDefault - Rotate\n" \
                     "\t\tCtrl - By Cursor\n" \
                     "\t\tShift - Individual\n" \
                     "\t\tAlt - CCW" \
                     "Has [5] keymap"
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode',
                       default='DEFAULT',
                       items=(('DEFAULT', 'Default', ''),
                              ('INDIVIDUAL', 'Individual', ''),
                              ('BY_CURSOR', 'By Cursor', ''))
                       )
    rot_dir: EnumProperty(name='Direction of rotation', default='CW', items=(('CW', 'CW', ''), ('CCW', 'CCW', '')))
    user_angle: FloatProperty(name='Angle', default=pi*0.5, min=0, max=pi, soft_min=math.radians(5.0), subtype='ANGLE')
    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(self, 'user_angle', slider=True)
        self.layout.prop(self, 'use_correct_aspect', toggle=1)
        self.layout.row(align=True).prop(self, 'rot_dir', expand=True)
        self.layout.row(align=True).prop(self, 'mode', expand=True)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            if context.area.type == 'IMAGE_EDITOR' and context.area.ui_type == 'UV':
                self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
                self.mouse_pos = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)

        self.rot_dir = 'CCW' if event.alt else 'CW'
        if event.shift:
            self.mode = 'INDIVIDUAL'
        elif event.ctrl:
            self.mode = 'BY_CURSOR'
        else:
            self.mode = 'DEFAULT'
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None
        self.angle = 0.0
        self.aspect = 1.0
        self.max_distance: float = 0.0
        self.mouse_pos: Vector | None = None

    def execute(self, context):
        self.angle = (-self.user_angle) if self.rot_dir == 'CCW' else self.user_angle
        self.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0

        self.umeshes = UMeshes(report=self.report)
        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if not self.umeshes:
            return self.umeshes.update()
        if not selected_umeshes and self.mouse_pos:
            return self.pick_rotate()

        if self.mode == 'DEFAULT':
            return self.rotate(extended=selected_umeshes)
        elif self.mode == 'BY_CURSOR':
            return self.rotate_by_cursor(extended=selected_umeshes)
        else:
            return self.rotate_individual(extended=selected_umeshes)

    def rotate(self, extended):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in self.umeshes:
            if islands := Islands.calc_extended_or_visible_with_mark_seam(umesh, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
            umesh.update_tag = bool(islands)

        pivot = general_bbox.center
        for islands in islands_of_mesh:
            islands.rotate(self.angle, pivot=pivot, aspect=self.aspect)
        return self.umeshes.update()

    def rotate_by_cursor(self, extended):
        if not (cursor := utils.get_cursor_location()):
            self.report({'INFO'}, "Cursor not found")
            return {'FINISHED'}
        for umesh in self.umeshes:
            if islands := Islands.calc_extended_or_visible_with_mark_seam(umesh, extended=extended):
                islands.rotate(self.angle, pivot=cursor, aspect=self.aspect)
            umesh.update_tag = bool(islands)
        return self.umeshes.update()

    def rotate_individual(self,  extended):
        for umesh in self.umeshes:
            if islands := Islands.calc_extended_or_visible_with_mark_seam(umesh, extended=extended):
                for island in islands:
                    island.rotate(self.angle, pivot=island.calc_bbox().center, aspect=self.aspect)
            umesh.update_tag = bool(islands)
        return self.umeshes.update()

    def pick_rotate(self):
        hit = types.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            for isl in types.AdvIslands.calc_visible_with_mark_seam(umesh):
                hit.find_nearest_island_by_crn(isl)

        if not hit:
            self.report({'INFO'}, 'Island not found within a given radius')
            return {'CANCELLED'}

        pivot = utils.get_cursor_location() if self.mode == 'BY_CURSOR' else hit.island.bbox.center
        hit.island.rotate(self.angle, pivot, self.aspect)
        hit.island.umesh.update()
        return {'FINISHED'}


class UNIV_OT_Sort(Operator, utils.OverlapHelper, utils.PaddingHelper):
    bl_idname = 'uv.univ_sort'
    bl_label = 'Sort'
    bl_description = \
        "Default - Sort islands\n" \
        "Shift - Lock Overlaps.\n" \
        "Ctrl - Start sort position to Cursor\n" \
        "Alt - Disable orient"
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(name='Axis', default='AUTO', items=(('AUTO', 'Auto', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    sub_padding: FloatProperty(name='Sub Padding', default=0.1, min=0, soft_max=0.2,)
    area_subgroups: IntProperty(name='Area Subgroups', default=4, min=1, max=200, soft_max=8)
    reverse: BoolProperty(name='Reverse', default=True)
    to_cursor: BoolProperty(name='To Cursor', default=False)
    orient: BoolProperty(name='Orient', default=False)
    subgroup_type: EnumProperty(name='Subgroup Type', default='NONE', items=(
        ('NONE', 'None', ''),
        ('AREA', 'Area', ''),
        ('OBJECTS', 'Objects', ''),
        ('MATERIALS', 'Materials', '')))

    def draw(self, context):
        layout = self.layout.row()
        layout.prop(self, 'axis', expand=True)
        layout = self.layout
        layout.prop(self, 'reverse')
        layout.prop(self, 'to_cursor')
        layout.prop(self, 'orient')
        if self.subgroup_type == 'NONE':
            self.draw_overlap()
        self.draw_padding()
        if self.subgroup_type != 'NONE':
            layout.prop(self, 'sub_padding', slider=True)
        if self.subgroup_type == 'AREA':
            layout.prop(self, 'area_subgroups')
        layout = self.layout.row()
        layout.prop(self, 'subgroup_type', expand=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.to_cursor = event.ctrl
        self.lock_overlap = event.shift
        self.orient = not event.alt
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync: bool = bpy.context.scene.tool_settings.use_uv_select_sync
        self.update_tag: bool = False
        self.cursor_loc: Vector | None = None
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        self.update_tag = False
        self.umeshes = UMeshes(report=self.report)
        self.calc_padding()
        self.report_padding()

        if self.to_cursor:
            if not (cursor_loc := utils.get_cursor_location()):
                self.report({'INFO'}, "Cursor not found")
                return {'CANCELLED'}
            self.cursor_loc = cursor_loc
        else:
            self.cursor_loc = None

        if self.subgroup_type != 'NONE':
            self.lock_overlap = False

        if not self.lock_overlap:
            self.sort_individual_preprocessing(extended=True)
            if not self.umeshes.final():
                self.sort_individual_preprocessing(extended=False)
        else:
            self.sort_overlapped_preprocessing(extended=True)
            if not self.umeshes.final():
                self.sort_overlapped_preprocessing(extended=False)

        if not self.update_tag:
            return self.umeshes.cancel_with_report(info='Islands is sorted')  # TODO: Add info when islands not found
        return self.umeshes.update()

    def sort_overlapped_preprocessing(self, extended=True):
        _islands: list[AdvIsland] = []
        for umesh in self.umeshes:
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh, extended=extended):
                adv_islands.calc_tris()
                adv_islands.calc_flat_coords()
                _islands.extend(adv_islands)
            umesh.update_tag = bool(adv_islands)

        if not _islands:
            return

        general_bbox = BBox()
        union_islands_groups = self.calc_overlapped_island_groups(_islands)
        for union_island in union_islands_groups:
            if self.orient:
                isl_coords = union_island.calc_convex_points()
                general_bbox.union(union_island.bbox)
                angle = utils.calc_min_align_angle(isl_coords)

                if not math.isclose(angle, 0, abs_tol=0.0001):
                    union_island.rotate_simple(angle)
                    union_island.calc_bbox()
            else:
                bb = union_island.bbox
                general_bbox.union(bb)

        is_horizontal = self.is_horizontal(general_bbox, union_islands_groups)
        margin = general_bbox.min if (self.cursor_loc is None) else self.cursor_loc
        self.sort_islands(is_horizontal, margin, union_islands_groups)

    def sort_individual_preprocessing(self, extended=True):
        _islands: list[AdvIsland] | list[AdvIslands] = []
        general_bbox = BBox()
        for umesh in self.umeshes:
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh, extended=extended):
                if self.orient:
                    for island in adv_islands:
                        isl_coords = island.calc_convex_points()
                        general_bbox.union(island.bbox)
                        angle = utils.calc_min_align_angle(isl_coords)
                        if not math.isclose(angle, 0, abs_tol=0.0001):
                            island.rotate_simple(angle)
                            island.calc_bbox()
                else:
                    for island in adv_islands:
                        general_bbox.union(island.bbox)
                if self.subgroup_type == 'OBJECTS':
                    _islands.append(adv_islands)  # noqa
                elif self.subgroup_type == 'MATERIALS':
                    adv_islands.calc_materials(umesh)
                    _islands.extend(adv_islands)
                elif self.subgroup_type == 'AREA':
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_coords()
                    adv_islands.calc_area_uv()
                    _islands.extend(adv_islands)
                else:
                    _islands.extend(adv_islands)

            umesh.update_tag = bool(adv_islands)

        if not _islands:
            return

        is_horizontal = self.is_horizontal(general_bbox, _islands)
        margin = general_bbox.min if (self.cursor_loc is None) else self.cursor_loc

        if self.subgroup_type == 'NONE':
            self.sort_islands(is_horizontal, margin, _islands)
        elif self.subgroup_type == 'OBJECTS':
            for islands in _islands:
                self.sort_islands(is_horizontal, margin, islands.islands)  # noqa
        elif self.subgroup_type == 'MATERIALS':
            subgroups = defaultdict(list)
            for island in _islands:
                subgroups[island.info.materials].append(island)  # noqa
            for islands in subgroups.values():
                self.sort_islands(is_horizontal, margin, islands)
        else:  # 'AREA'
            subgroups = self.calc_area_subgroups(_islands)
            if self.reverse:
                for islands in reversed(subgroups):
                    self.sort_islands(is_horizontal, margin, islands)
            else:
                for islands in subgroups:
                    self.sort_islands(is_horizontal, margin, islands)

    def calc_area_subgroups(self, islands: list[AdvIsland]):
        islands.sort(reverse=True, key=lambda a: a.area_uv)
        splitted = []
        if len(islands) > 1:
            start = islands[0].area_uv
            end = islands[-1].area_uv
            segment = (start - end) / self.area_subgroups
            end += 0.00001

            for i in range(self.area_subgroups):
                seg = []
                end += segment
                if not islands:
                    break

                for j in range(len(islands) - 1, -1, -1):
                    if islands[j].area_uv <= end:
                        seg.append(islands.pop())
                    else:
                        break
                if seg:
                    splitted.append(seg)

            assert (not islands), 'Extremal Values'
        else:
            splitted = [islands]
        return splitted

    def sort_islands(self, is_horizontal: bool, margin: Vector, islands: list[AdvIsland | UnionIslands] | AdvIslands):
        islands.sort(key=lambda x: x.bbox.max_length, reverse=self.reverse)
        if is_horizontal:
            for island in islands:
                width = island.bbox.width
                if self.orient and island.bbox.height < width:
                    width = island.bbox.height
                    self.update_tag |= island.rotate(pi * 0.5, island.bbox.center)
                    island.calc_bbox()
                self.update_tag |= island.set_position(margin, _from=island.bbox.min)
                margin.x += self.padding + width
            margin.x += self.sub_padding
        else:
            for island in islands:
                height = island.bbox.height
                if self.orient and island.bbox.width < height:
                    height = island.bbox.width
                    self.update_tag |= island.rotate(pi * 0.5, island.bbox.center)
                    island.calc_bbox()  # TODO: Optimize this
                self.update_tag |= island.set_position(margin, _from=island.bbox.min)
                margin.y += self.padding + height
            margin.y += self.sub_padding

    def is_horizontal(self, bbox, islands):
        if self.axis == 'AUTO':
            if bbox.width * 1.5 > bbox.height:
                return True
            else:
                total_width = 0
                total_height = 0
                if type(islands[0]) == AdvIslands:
                    islands = (isl_ for _islands in islands for isl_ in _islands)

                for isl in islands:
                    bbox_ = isl.bbox
                    total_width += bbox_.width
                    total_height += bbox_.height
                return total_width < total_height
        else:
            return self.axis == 'X'


class UNIV_OT_Distribute(Operator, utils.OverlapHelper, utils.PaddingHelper):
    bl_idname = 'uv.univ_distribute'
    bl_label = 'Distribute'
    bl_description = "Distribute\n\n" \
                     "Context keymaps on button:\n" \
                     "\t\tDefault - Distribute\n" \
                     "\t\tCtrl - To Cursor\n" \
                     "\t\tShift - Overlapped\n" \
                     "\t\tAlt - Break"
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(name='Axis', default='AUTO', items=(('AUTO', 'Auto', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    space: EnumProperty(name='Space', default='ALIGN', items=(('ALIGN', 'Align', ''), ('SPACE', 'Space', '')),
                        description='Distribution of islands at equal distances')
    to_cursor: BoolProperty(name='To Cursor', default=False)
    break_: BoolProperty(name='Break', default=False)
    angle: FloatProperty(name='Smooth Angle', default=math.radians(66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))

    def draw(self, context):
        if not self.break_:
            layout = self.layout.row()
            layout.prop(self, 'space', expand=True)
            layout = self.layout
            self.draw_overlap()
            layout.prop(self, 'to_cursor')
            layout.prop(self, 'break_')
        else:
            layout = self.layout.row()
            layout.prop(self, 'break_')
            layout.prop(self, 'angle', slider=True)

        layout = self.layout.row()
        layout.prop(self, 'axis', expand=True)
        self.draw_padding()

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.to_cursor = event.ctrl
        self.lock_overlap = event.shift
        self.break_ = event.alt
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync = bpy.context.scene.tool_settings.use_uv_select_sync
        self.umeshes: types.UMeshes | None = None
        self.cursor_loc: Vector | None = None
        self.update_tag = False

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
        self.calc_padding()
        self.report_padding()

        if self.to_cursor and not self.break_:
            if not (cursor_loc := utils.get_cursor_location()):
                self.report({'INFO'}, "Cursor not found")
                return {'CANCELLED'}
            self.cursor_loc = cursor_loc
        else:
            self.cursor_loc = None

        if self.break_:
            max_angle = max(umesh.smooth_angle for umesh in self.umeshes)
            self.angle = min(self.angle, max_angle)  # clamp for angle

            self.distribute_break_preprocessing(extended=True)
            if not self.umeshes.final():
                self.distribute_break_preprocessing(extended=False)
        elif self.space == 'SPACE':
            self.distribute_space(extended=True)
            if not self.umeshes.final():
                self.distribute_space(extended=False)
        else:
            self.distribute(extended=True)
            if not self.umeshes.final():
                self.distribute(extended=False)
        self.umeshes.update()
        return {'FINISHED'}

    def distribute_break_preprocessing(self, extended):
        cancel = False
        for umesh in self.umeshes:
            self.update_tag = False
            angle = min(self.angle, umesh.smooth_angle)
            umesh.value = angle
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh, extended=extended):
                for isl in adv_islands:
                    if len(isl) == 1:
                        continue
                    sub_islands = isl.calc_sub_islands_all()
                    if len(sub_islands) > 1:
                        self.distribute_ex(list(sub_islands), isl.bbox)
            umesh.update_tag = self.update_tag
            cancel |= bool(adv_islands)
        if cancel and not any(umesh.update_tag for umesh in self.umeshes):
            self.umeshes.cancel_with_report(info=f"Islands for break not found")

    def distribute_ex(self, _islands, general_bbox):
        if len(_islands) < 2:
            if len(_islands) == 1:
                self.umeshes.cancel_with_report(info=f"The number of islands must be greater than one")
            return

        cursor_offset = 0
        if self.is_horizontal(general_bbox, _islands):
            _islands.sort(key=lambda a: a.bbox.xmin)
            if self.cursor_loc is None:
                margin = general_bbox.min.x
            else:
                margin = self.cursor_loc.x
                cursor_offset += general_bbox.min.y - self.cursor_loc.y

            for island in _islands:
                width = island.bbox.width
                self.update_tag |= island.set_position(Vector((margin, island.bbox.ymin - cursor_offset)), _from=island.bbox.min)
                margin += self.padding + width
        else:
            _islands.sort(key=lambda a: a.bbox.ymin)
            if self.cursor_loc is None:
                margin = general_bbox.min.y
            else:
                margin = self.cursor_loc.y
                cursor_offset += general_bbox.min.x - self.cursor_loc.x

            for island in _islands:
                height = island.bbox.height
                self.update_tag |= island.set_position(Vector((island.bbox.xmin - cursor_offset, margin)), _from=island.bbox.min)
                margin += self.padding + height

    def distribute(self, extended=True):
        self.update_tag = False
        if self.lock_overlap:
            func = self.distribute_preprocessing_overlap
        else:
            func = self.distribute_preprocessing
        self.distribute_ex(*func(extended))

        if not self.update_tag and any(umesh.update_tag for umesh in self.umeshes):
            self.umeshes.cancel_with_report(info='Islands is Distributed')

    def distribute_space(self, extended=True):
        if self.lock_overlap:
            func = self.distribute_preprocessing_overlap
        else:
            func = self.distribute_preprocessing
        _islands, general_bbox = func(extended)

        if len(_islands) <= 2:
            if len(_islands) != 0:
                self.umeshes.cancel_with_report(info=f"The number of islands must be greater than two, {len(_islands)} was found")
            return

        update_tag = False
        cursor_offset = 0
        if self.is_horizontal(general_bbox, _islands):
            _islands.sort(key=lambda a: a.bbox.xmin)

            general_bbox.xmax += self.padding * (len(_islands) - 1)
            start_space = general_bbox.xmin + _islands[0].bbox.half_width
            end_space = general_bbox.xmax - _islands[-1].bbox.half_width
            if start_space == end_space:
                self.umeshes.cancel_with_report(info=f"No distance to place UV")
                return

            if self.cursor_loc:
                diff = end_space - start_space
                start_space += self.cursor_loc.x - start_space
                end_space = start_space + diff
                cursor_offset += general_bbox.ymin - self.cursor_loc.y
            space_points = np.linspace(start_space, end_space, len(_islands))

            for island, space_point in zip(_islands, space_points):
                update_tag |= island.set_position(Vector((space_point, island.bbox.center_y - cursor_offset)), _from=island.bbox.center)
        else:
            _islands.sort(key=lambda a: a.bbox.ymin)
            general_bbox.ymax += self.padding * (len(_islands) - 1)
            start_space = general_bbox.ymin + _islands[0].bbox.half_height
            end_space = general_bbox.ymax - _islands[-1].bbox.half_height
            if start_space == end_space:
                self.umeshes.cancel_with_report(info=f"No distance to place UV")
                return
            if self.cursor_loc:
                start_space += start_space - self.cursor_loc.y
                end_space += end_space - self.cursor_loc.y

            if self.cursor_loc:
                diff = end_space - start_space
                start_space += self.cursor_loc.y - start_space
                end_space = start_space + diff
                cursor_offset += general_bbox.xmin - self.cursor_loc.x

            space_points = np.linspace(start_space, end_space, len(_islands))

            for island, space_point in zip(_islands, space_points):
                update_tag |= island.set_position(Vector((island.bbox.center_x - cursor_offset, space_point)), _from=island.bbox.center)

        if not update_tag:
            self.umeshes.cancel_with_report(info='Islands is Distributed')

    def distribute_preprocessing(self, extended):
        _islands: list[AdvIsland] = []
        general_bbox = BBox()
        for umesh in self.umeshes:
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh, extended=extended):
                general_bbox.union(adv_islands.calc_bbox())
                _islands.extend(adv_islands)
            umesh.update_tag = bool(adv_islands)
        return _islands, general_bbox

    def distribute_preprocessing_overlap(self, extended):
        _islands: list[AdvIsland] = []
        for umesh in self.umeshes:
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh, extended=extended):
                adv_islands.calc_tris()
                adv_islands.calc_flat_coords()
                _islands.extend(adv_islands)
            umesh.update_tag = bool(adv_islands)

        general_bbox = BBox()
        union_islands_groups = self.calc_overlapped_island_groups(_islands)
        for union_island in union_islands_groups:
            general_bbox.union(union_island.bbox)
        return union_islands_groups, general_bbox

    def is_horizontal(self, bbox, islands):
        if self.axis == 'AUTO':
            if self.break_:
                return bbox.width > bbox.height
            else:
                total_width = 0
                total_height = 0
                if type(islands[0]) == AdvIslands:
                    islands = (isl_ for _islands in islands for isl_ in _islands)

                for isl in islands:
                    bbox_ = isl.bbox
                    total_width += bbox_.width
                    total_height += bbox_.height
                return total_width < total_height
        else:
            return self.axis == 'X'


class UNIV_OT_Home(Operator):
    bl_idname = 'uv.univ_home'
    bl_label = 'Home'
    bl_description = "Move island to base tile without changes in the textured object\n\n" \
                     "Default - Move island to base tile\n" \
                     "Ctrl - Move island to cursor.\n\n" \
                     "Removes attributes and modifiers from Shift operator, " \
                     "resets uv offset of Array, Mirror, UVWarp modifiers"
    bl_options = {'REGISTER', 'UNDO'}

    to_cursor: BoolProperty(name='To Cursor', default=False)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.to_cursor = event.ctrl
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_selected = True
        self.gn_mod_counter = 0
        self.shift_attrs_counter = 0
        self.islands_calc_type: Callable = Callable
        self.umeshes: types.UMeshes | None = None
        self.no_change_info = "Not found islands for move and modifiers for reset uv offsets"

    def execute(self, context):
        self.gn_mod_counter = 0
        self.shift_attrs_counter = 0

        cursor_loc = Vector((0, 0))
        if self.to_cursor and not (cursor_loc := utils.get_tile_from_cursor()):
            self.report({'WARNING'}, "Cursor not found")
            return {'CANCELLED'}

        self.umeshes = types.UMeshes.calc_any_unique(verify_uv=False)

        mod_counter, attr_counter = self.remove_shift_md()  # remove_shift_md changes update tag
        changed_modifiers_count = self.uv_shift_reset_array_and_mirror_and_warp()
        changed_modifiers_count += mod_counter

        report_info = ''
        if changed_modifiers_count:
            report_info += f"Changed {changed_modifiers_count} modifiers."
        if attr_counter:
            report_info += f"Deleted {attr_counter} shift attributes."

        self.umeshes = types.UMeshes()
        self.umeshes.update_tag = False
        if self.umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
        else:
            selected_umeshes = self.umeshes
            unselected_umeshes = []

        if selected_umeshes:
            self.umeshes = selected_umeshes
            self.islands_calc_type = AdvIslands.calc_extended_with_mark_seam
        elif unselected_umeshes:
            self.umeshes = unselected_umeshes
            self.islands_calc_type = AdvIslands.calc_visible_with_mark_seam
        else:
            if report_info:
                self.report({'INFO'}, report_info)
                return {'FINISHED'}
            else:
                self.report({'INFO'}, self.no_change_info)
                return {'CANCELLED'}

        if not self.umeshes.is_edit_mode:
            self.islands_calc_type = AdvIslands.calc_with_hidden_with_mark_seam

        counter = 0
        for umesh in self.umeshes:
            for island in self.islands_calc_type(umesh):  # noqa
                counter += self.home(island, cursor_loc)

        if counter or report_info:
            if report_info:
                self.report({'INFO'}, report_info)
        else:
            self.report({'INFO'}, self.no_change_info)
        self.umeshes.silent_update()
        self.umeshes.free()
        return {'FINISHED'}

    @staticmethod
    def home(island, cursor):
        center = island.calc_bbox().center
        delta = Vector(round(-i + 0.5) for i in center) + cursor
        tag = island.move(delta)
        island.umesh.update_tag |= tag
        return tag

    def remove_shift_md(self):
        all_object = set(obj for obj in bpy.data.objects if obj.type == 'MESH')
        all_object = all_object - set(umesh.obj for umesh in self.umeshes)
        mod_counter = 0
        attr_counter = 0
        for umesh in self.umeshes:
            for mod in reversed(umesh.obj.modifiers):
                if isinstance(mod, bpy.types.NodesModifier) and mod.name.startswith('UniV Shift'):
                    umesh.obj.modifiers.remove(mod)
                    mod_counter += 1

            # safe attr for instances if not zero
            instances = (inst_obj for inst_obj in all_object if inst_obj.data == umesh.obj.data)
            has_inst_with_shift_mod = False
            for inst_obj in instances:
                for mod in inst_obj.modifiers:
                    if isinstance(mod, bpy.types.NodesModifier) and mod.name.startswith('UniV Shift'):
                        has_inst_with_shift_mod = True
                        break

            if not has_inst_with_shift_mod:
                if self.umeshes.is_edit_mode:
                    for attr in reversed(umesh.bm.faces.layers.int.values()):
                        if attr.name.startswith('univ_shift'):
                            umesh.bm.faces.layers.int.remove(attr)
                            umesh.update_tag = True
                            attr_counter += 1
                else:
                    for attr in reversed(umesh.obj.data.attributes):
                        if attr.name.startswith('univ_shift'):
                            umesh.obj.data.attributes.remove(attr)
                            umesh.update_tag = True
                            attr_counter += 1

        # Remove modifiers without univ_shift attr
        for other_umesh in all_object:
            if any(other_umesh.data == umesh.obj.data for umesh in self.umeshes):
                attributes = other_umesh.data.attributes
                if not any(attr.name.startswith('univ_shift') for attr in attributes):
                    for mod in reversed(other_umesh.modifiers):
                        if isinstance(mod, bpy.types.NodesModifier) and mod.name.startswith('UniV Shift'):
                            other_umesh.modifiers.remove(mod)
                            mod_counter += 1

        return mod_counter, attr_counter

    def uv_shift_reset_array_and_mirror_and_warp(self):
        counter = 0
        for umesh in self.umeshes:
            for mod in umesh.obj.modifiers:
                if isinstance(mod, bpy.types.ArrayModifier):
                    if any((mod.offset_u, mod.offset_v)):
                        mod.offset_u = 0.0
                        mod.offset_v = 0.0
                        counter += 1
                elif isinstance(mod, bpy.types.MirrorModifier):
                    if any((mod.offset_u, mod.offset_v)):
                        mod.offset_u = 0.0
                        mod.offset_v = 0.0
                        counter += 1
                elif isinstance(mod, bpy.types.UVWarpModifier):
                    if any(mod.offset):
                        mod.offset[0] = 0.0
                        mod.offset[1] = 0.0
                        counter += 1
        return counter


class UNIV_OT_Shift(Operator):
    bl_idname = "uv.univ_shift"
    bl_label = 'Shift'
    bl_description = "Moving overlapped islands to an adjacent tile, to avoid artifacts when baking"
    bl_options = {'REGISTER', 'UNDO'}

    lock_overlap_mode: EnumProperty(name='Lock Overlaps Mode', default='ANY', items=(('ANY', 'Any', ''), ('EXACT', 'Exact', '')))
    threshold: FloatProperty(name='Distance', default=0.001, min=0, soft_min=0.00005, soft_max=0.00999)
    shift_smaller: BoolProperty(name='Shift Smaller', default=False, description="Sets a higher priority for shifting, for small islands")

    with_modifier: BoolProperty(name='Use Modifiers', default=False,
                                description="Non-destructively through a modifier shifts islands. To remove a modifier, use the Home operator.")
    gn_shift: BoolProperty(name='GN Shift', default=True, description="Add Shift Geometry Node Modifier.")
    array_shift: BoolProperty(name='Array', default=True, description='U Offset')
    mirror_shift: BoolProperty(name='Mirror', default=True, description='U Offset')
    warp_shift: BoolProperty(name='Warp', default=True, description='U Offset')

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.with_modifier = event.alt
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        if self.lock_overlap_mode == 'EXACT':
            layout.prop(self, 'threshold', slider=True)
        else:
            layout.prop(self, 'shift_smaller')
        layout.row().prop(self, 'lock_overlap_mode', expand=True)
        if self.with_modifier:
            row = layout.row(align=True)
            row.prop(self, 'gn_shift', toggle=1)
            row.prop(self, 'array_shift', toggle=1)
            row.prop(self, 'mirror_shift', toggle=1)
            row.prop(self, 'warp_shift', toggle=1)
        layout.prop(self, 'with_modifier', toggle=1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_selected = True
        self.islands_calc_type: Callable = Callable
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.umeshes = types.UMeshes.calc_any_unique(verify_uv=False)
        changed_modifiers = self.shift_array_and_mirror_and_warp()

        # TODO: Remove gn modifier when shift without modifier
        self.umeshes = types.UMeshes()
        if self.umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
        else:
            selected_umeshes = self.umeshes
            unselected_umeshes = []

        if selected_umeshes:
            self.has_selected = True
            self.umeshes = selected_umeshes

            self.islands_calc_type = AdvIslands.calc_extended_with_mark_seam
        elif unselected_umeshes:
            self.has_selected = False
            self.umeshes = unselected_umeshes
            self.islands_calc_type = AdvIslands.calc_visible_with_mark_seam
        else:
            if changed_modifiers:
                self.report({'INFO'}, f"Changed {changed_modifiers} modifiers")
                return
            else:
                self.report({'WARNING'}, 'Islands not found')
                return {'CANCELLED'}

        if not self.umeshes.is_edit_mode:
            self.islands_calc_type = AdvIslands.calc_with_hidden_with_mark_seam

        umeshes_without_attributes = []
        if self.with_modifier and self.gn_shift:
            for umesh in self.umeshes:
                if 'univ_shift' not in umesh.obj.data.attributes:
                    umesh.bm.faces.layers.int.new('univ_shift')
                    umeshes_without_attributes.append(umesh)

        all_islands = []
        self.umeshes.update_tag = False

        for umesh in self.umeshes:
            adv_islands = self.islands_calc_type(umesh)  # noqa
            for isl in reversed(adv_islands):
                if isl.has_flip_with_noflip():
                    adv_islands.islands.remove(isl)
                    noflip, flipped = isl.calc_islands_by_flip_with_mark_seam()
                    adv_islands.islands.extend(noflip)
                    adv_islands.islands.extend(flipped)  # TODO: Add info about flip and no flip

            if self.lock_overlap_mode == 'ANY':
                adv_islands.calc_tris()
                adv_islands.calc_flat_uv_coords(save_triplet=True)
            all_islands.extend(adv_islands)

        counter = 0
        deleted_attr_counter = 0
        threshold = None if self.lock_overlap_mode == 'ANY' else self.threshold

        overlapped_islands = types.UnionIslands.calc_overlapped_island_groups(all_islands, threshold)
        for over_isl in reversed(overlapped_islands):
            if isinstance(over_isl, AdvIsland) or len(over_isl) == 1:
                overlapped_islands.remove(over_isl)
                continue

            if self.shift_smaller and self.lock_overlap_mode == 'ANY':
                for sub_isl in over_isl:
                    sub_isl.calc_area_uv()
                f_island = over_isl.islands[0]
                index_for_bigger = 0
                for idx_, sub_isl in enumerate(over_isl):
                    if isclose(f_island.area_uv, sub_isl.area_uv, abs_tol=1e-04):
                        continue
                    if sub_isl.area_uv > f_island.area_uv:
                        f_island = sub_isl
                        index_for_bigger = idx_
                if index_for_bigger:
                    over_isl[0], over_isl[index_for_bigger] = over_isl[index_for_bigger], over_isl[0]

            for idx, isl in enumerate(over_isl):
                isl.umesh.value = 1
                if self.with_modifier and self.gn_shift:
                    if idx:
                        shift_attr = isl.umesh.bm.faces.layers.int.get('univ_shift')
                        if not all(f[shift_attr] for f in isl):
                            for f in isl:
                                f[shift_attr] = 1

                            isl.umesh.update_tag = True
                            counter += 1
                    else:
                        if 'univ_shift' in isl.umesh.obj.data.attributes:
                            shift_attr = isl.umesh.bm.faces.layers.int.get('univ_shift')
                            if any(f[shift_attr] for f in isl):
                                for f in isl:
                                    f[shift_attr] = 0
                                isl.umesh.update_tag = True
                else:
                    if idx:
                        counter += 1
                        isl.umesh.update_tag = True
                        isl.move(Vector((1.0, 0.0)))

        if self.with_modifier and self.gn_shift:
            node_group = self.get_shift_node_group()
            for umesh in self.umeshes:
                if umesh.update_tag or isinstance(umesh.value, int):
                    utils.remove_univ_duplicate_modifiers(umesh.obj, 'UniV Shift')
                    self.create_gn_shift_modifier(umesh, node_group)

        # Sanitize
        for umesh in self.umeshes:
            for attr in reversed(umesh.bm.faces.layers.int.values()):
                if attr.name.startswith('univ_shift'):
                    if not any(f[attr] for f in umesh.bm.faces):
                        umesh.bm.faces.layers.int.remove(attr)
                        umesh.update_tag = True
                        if umesh not in umeshes_without_attributes:
                            deleted_attr_counter += 1
            if not any(attr.name.startswith('univ_shift') for attr in umesh.bm.faces.layers.int.values()):
                for mod in reversed(umesh.obj.modifiers):
                    if isinstance(mod, bpy.types.NodesModifier) and mod.name.startswith('UniV Shift'):
                        umesh.obj.modifiers.remove(mod)
                        changed_modifiers += 1

        report_info = ''
        if changed_modifiers:
            report_info += f"Changed {changed_modifiers} modifiers."
        if deleted_attr_counter:
            report_info += f"Deleted {deleted_attr_counter} unused attributes."

        if report_info:
            self.report({'INFO'}, report_info)
        elif not counter:
            self.report({'WARNING'}, 'Not found islands and modifiers for shift')

        self.umeshes.silent_update()
        self.umeshes.free()
        return {'FINISHED'}

    def shift_array_and_mirror_and_warp(self):
        if not self.with_modifier:
            return 0
        counter = 0
        for umesh in self.umeshes:
            for mod in umesh.obj.modifiers:
                if self.array_shift and isinstance(mod, bpy.types.ArrayModifier):
                    if mod.offset_u != 1.0:
                        mod.offset_u = 1.0
                        counter += 1
                elif self.mirror_shift and isinstance(mod, bpy.types.MirrorModifier):
                    if mod.offset_u != 1.0:
                        mod.offset_u = 1.0
                        counter += 1
                elif self.warp_shift and isinstance(mod, bpy.types.UVWarpModifier):
                    if mod.offset[0] != 1.0:
                        mod.offset[0] = 1.0
                        counter += 1
        return counter

    @staticmethod
    def shift_node_group_is_changed(node_group):
        if len(nodes := node_group.nodes) != 7:
            return True

        if not (output_node := [n for n in nodes if n.bl_idname == 'NodeGroupOutput']) or \
                not output_node[0].inputs or not (output_links := output_node[0].inputs[0].links):
            return True

        if (store_attr_node := output_links[0].from_node).bl_idname != 'GeometryNodeStoreNamedAttribute' or \
                store_attr_node.data_type != 'FLOAT2' and store_attr_node.domain != 'CORNER':
            return True

        if not (store_attr_node_geometry_links := store_attr_node.inputs[0].links) or \
                (store_attr_node_geometry_links[0].from_node.bl_idname != 'NodeGroupInput'):
            return True

        if not (store_attr_node_name_links := store_attr_node.inputs['Name'].links) or \
                store_attr_node_name_links[0].from_node.bl_idname != 'NodeGroupInput':
            return True

        if not (store_attr_node_value_links := store_attr_node.inputs['Value'].links) or \
                (vector_node := store_attr_node_value_links[0].from_node).bl_idname != 'ShaderNodeVectorMath':
            return True

        if vector_node.operation != 'ADD':
            return True

        if not (vector_node_a_links := vector_node.inputs[0].links) or not (vector_node_b_links := vector_node.inputs[1].links):
            return True

        if (uvmap_node := vector_node_a_links[0].from_node).bl_idname != 'GeometryNodeInputNamedAttribute' or \
                not (uvmap_name_links := uvmap_node.inputs['Name'].links) or \
                uvmap_name_links[0].from_node.bl_idname != 'NodeGroupInput' or uvmap_node.data_type != 'FLOAT_VECTOR':  # noqa
            return True

        if (combine_xyz_node := vector_node_b_links[0].from_node).bl_idname != 'ShaderNodeCombineXYZ' or \
                not (x_links := combine_xyz_node.inputs['X'].links) or \
                (shift_node := x_links[0].from_node).bl_idname != 'GeometryNodeInputNamedAttribute':  # noqa
            return True

        if shift_node.data_type != 'BOOLEAN' or shift_node.inputs['Name'].default_value != 'univ_shift':
            return True

        return False

    @staticmethod
    def create_shift_node_group():
        node_group = bpy.data.node_groups.new(name='UniV Shift', type='GeometryNodeTree')

        create_node = node_group.nodes.new
        input_node = create_node(type="NodeGroupInput")
        input_node.location = (-800, -80)

        output_node = create_node(type="NodeGroupOutput")
        output_node.location = (200, 0)

        store_attr_node = create_node(type="GeometryNodeStoreNamedAttribute")
        store_attr_node.location = (0, 0)
        store_attr_node.data_type = 'FLOAT2'
        store_attr_node.domain = 'CORNER'

        shift_node = create_node(type="GeometryNodeInputNamedAttribute")
        shift_node.location = (-600, -330)
        shift_node.data_type = 'BOOLEAN'
        shift_node.inputs['Name'].default_value = 'univ_shift'

        uvmap_node = create_node(type="GeometryNodeInputNamedAttribute")
        uvmap_node.location = (-600, -180)
        uvmap_node.data_type = 'FLOAT_VECTOR'

        combine_xyz_node = create_node(type="ShaderNodeCombineXYZ")
        combine_xyz_node.location = (-380, -330)

        vector_add_node = create_node(type="ShaderNodeVectorMath")
        vector_add_node.location = (-180, -180)

        if iface := getattr(node_group, 'interface', None):
            iface.new_socket('Input', description="", in_out='INPUT', socket_type='NodeSocketGeometry')
            iface.new_socket('UVMap', description="", in_out='INPUT', socket_type='NodeSocketString')
            iface.new_socket('Output', description="", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        else:
            node_group.inputs.new('NodeSocketGeometry', 'Input')
            node_group.inputs.new('NodeSocketString', 'UVMap')
            node_group.outputs.new('NodeSocketGeometry', 'Output')

        link = node_group.links.new

        link(input_node.outputs['Input'], store_attr_node.inputs['Geometry'])
        link(store_attr_node.outputs['Geometry'], output_node.inputs['Output'])
        link(input_node.outputs['UVMap'], store_attr_node.inputs['Name'])
        link(input_node.outputs['UVMap'], uvmap_node.inputs['Name'])

        for attr_output in shift_node.outputs:
            if attr_output.name == 'Attribute' and not attr_output.is_unavailable:
                link(attr_output, combine_xyz_node.inputs[0])
                break
        link(combine_xyz_node.outputs['Vector'], vector_add_node.inputs[1])

        for attr_output in uvmap_node.outputs:
            if attr_output.name == 'Attribute' and not attr_output.is_unavailable:
                link(attr_output, vector_add_node.inputs[0])
                break
        link(vector_add_node.outputs['Vector'], store_attr_node.inputs['Value'])

        return node_group

    @staticmethod
    def create_gn_shift_modifier(umesh, node_group):
        has_checker_modifier = False
        uv_name = umesh.uv.name
        for m in umesh.obj.modifiers:
            if not isinstance(m, bpy.types.NodesModifier):
                continue
            if m.name.startswith('UniV Shift'):
                has_checker_modifier = True
                if m.node_group != node_group:
                    m.node_group = node_group
                if 'Socket_1' in m:
                    if m['Socket_1'] != uv_name:
                        m['Socket_1'] = uv_name
                else:
                    # old version support (version???)
                    if m['Input_1'] != uv_name:
                        m['Input_1'] = uv_name
                umesh.update_tag = True
                break
        if not has_checker_modifier:
            m = umesh.obj.modifiers.new(name='UniV Shift', type='NODES')
            m.node_group = node_group
            if 'Socket_1' in m:
                m['Socket_1'] = uv_name
            else:
                m['Input_1'] = uv_name
            umesh.update_tag = True

    def get_shift_node_group(self):
        """Get exist checker material"""
        for ng in reversed(bpy.data.node_groups):
            if ng.name.startswith('UniV Shift'):
                if self.shift_node_group_is_changed(ng):
                    if ng.users == 0:
                        bpy.data.node_groups.remove(ng)
                else:
                    return ng
        return self.create_shift_node_group()


class UNIV_OT_Random(Operator, utils.OverlapHelper):
    bl_idname = "uv.univ_random"
    bl_label = "Random"
    bl_description = "Randomize selected UV islands or faces"
    bl_options = {'REGISTER', 'UNDO'}

    between: BoolProperty(name='Shaffle', default=False)
    bound_between: EnumProperty(name='Bound Shaffle', default='OFF', items=(('OFF', 'Off', ''), ('CROP', 'Crop', ''), ('CLAMP', 'Clamp', '')))
    round_mode: EnumProperty(name='Round Mode', default='OFF', items=(('OFF', 'Off', ''), ('INT', 'Int', ''), ('STEPS', 'Steps', '')))
    steps: FloatVectorProperty(name='Steps', description="Incorrectly works with Within Image Bounds",
                               default=(0, 0), min=0, max=10, soft_min=0, soft_max=1, size=2, subtype='XYZ')
    strength: FloatVectorProperty(name='Strength', default=(1, 1), min=-10, max=10, soft_min=0, soft_max=1, size=2, subtype='XYZ')
    flip_strength: FloatVectorProperty(name='Flip', default=(0, 0), min=0, max=1, size=2, subtype='XYZ')
    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)
    rotation: FloatProperty(name='Rotation Range', default=0, min=0, soft_max=math.pi * 2, subtype='ANGLE',
        update=lambda self, _: setattr(self, 'rotation_steps', self.rotation) if self.rotation < self.rotation_steps else None)
    rotation_steps: FloatProperty(name='Rotation Steps', default=0, min=0, max=math.pi, subtype='ANGLE',
        update=lambda self, _: setattr(self, 'rotation', self.rotation_steps) if self.rotation < self.rotation_steps else None)
    scale_factor: FloatProperty(name="Scale Factor", default=0, min=0, soft_max=1, subtype='FACTOR')
    min_scale: FloatProperty(name='Min Scale', default=0.5, min=0, max=10, soft_min=0.1, soft_max=2,
                             update=lambda self, _: setattr(self, 'max_scale', self.min_scale) if self.max_scale < self.min_scale else None)
    max_scale: FloatProperty(name='Max Scale', default=2, min=0, max=10, soft_min=0.1, soft_max=2,
                             update=lambda self, _: setattr(self, 'min_scale', self.max_scale) if self.max_scale < self.min_scale else None)
    bool_bounds: BoolProperty(name="Within Image Bounds", default=False, description="Keep the UV faces/islands within the 0-1 UV domain.", )
    rand_seed: IntProperty(name='Seed', default=0)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'rand_seed')

        if not self.between:
            layout.prop(self, 'round_mode', slider=True)
            if self.round_mode == 'STEPS':
                layout.prop(self, 'steps', slider=True)
            layout.prop(self, 'strength', slider=True)
        layout.prop(self, 'flip_strength', slider=True)

        if self.bound_between != 'CROP':
            layout.prop(self, 'scale_factor', slider=True)
            if self.scale_factor != 0:
                layout.prop(self, 'min_scale', slider=True)
                layout.prop(self, 'max_scale', slider=True)

        layout.prop(self, 'rotation', slider=True)
        if self.rotation != 0:
            layout.prop(self, 'rotation_steps', slider=True)
            layout.prop(self, 'use_correct_aspect', toggle=1)  # TODO: Implement for crop and clamp

        if not self.between:
            layout.prop(self, 'bool_bounds')

        layout = self.layout.row()
        if self.between:
            layout.prop(self, 'bound_between', expand=True)
        layout = self.layout.row()
        self.draw_overlap()
        layout.prop(self, 'between', toggle=1)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.lock_overlap = event.shift
        self.between = event.alt
        self.bound_between = 'CROP' if event.ctrl else 'OFF'
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = 1000
        self.aspect = 1.0
        self.non_valid_counter = 0
        self.umeshes: types.UMeshes | None = None
        self.is_edit_mode: bool = bpy.context.mode == 'EDIT_MESH'
        self.all_islands: list[UnionIslands | AdvIsland] | None = None
        self.sync = bpy.context.scene.tool_settings.use_uv_select_sync

    def execute(self, context):
        self.non_valid_counter = 0
        self.umeshes = types.UMeshes(report=self.report)
        if not self.umeshes:
            self.report({'WARNING'}, 'Objects not found')
            return {'CANCELLED'}

        self.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0

        if not self.is_edit_mode:
            self.umeshes.ensure(face=True)

        self.random_preprocessing()
        if self.is_edit_mode:
            if not self.all_islands:
                self.random_preprocessing(extended=False)

        if self.between:
            self.random_between()
        else:
            self.random()
        if self.non_valid_counter:
            self.report({'INFO'}, f"Found {self.non_valid_counter} zero-sized islands that will not be affected by some effects")
        self.umeshes.update(info="No object for randomize.")

        if not self.is_edit_mode:
            self.umeshes.free()
            utils.update_area_by_type('VIEW_3D')
        return {'FINISHED'}

    def random_preprocessing(self, extended=True):
        self.seed = sum(id(umesh.obj) for umesh in self.umeshes) // len(self.umeshes) + self.rand_seed

        self.all_islands = []
        _islands = []
        for umesh in self.umeshes:
            if self.is_edit_mode:
                islands = AdvIslands.calc_extended_or_visible(umesh, extended=extended)
            else:
                islands = AdvIslands.calc_with_hidden(umesh)
            if islands:
                if self.lock_overlap:
                    islands.calc_tris()
                    islands.calc_flat_coords()
                    _islands.extend(islands)
                else:
                    self.all_islands.extend(islands)
            umesh.update_tag = bool(islands)

        if self.lock_overlap:
            self.all_islands = self.calc_overlapped_island_groups(_islands)

    def random(self):
        for e_seed, island in enumerate(self.all_islands, start=100):
            seed = e_seed + self.seed
            random.seed(seed)
            rand_rotation = random.uniform(-self.rotation, self.rotation)
            random.seed(seed + 1000)
            rand_scale = random.uniform(self.min_scale, self.max_scale)
            flip_x = random.choices([-1, 1], weights=[self.flip_strength[0], 1 - self.flip_strength[0]], k=1)[0]
            flip_y = random.choices([-1, 1], weights=[self.flip_strength[1], 1 - self.flip_strength[1]], k=1)[0]

            if (self.bool_bounds or self.rotation or
                    self.scale_factor != 0 or -1 in (flip_x, flip_y)):

                bb = island.bbox
                if bb.min_length == 0:
                    self.non_valid_counter += 1
                    continue

                vec_origin = bb.center
                if -1 in (flip_x, flip_y):
                    island.scale(Vector((flip_x, flip_y)), pivot=vec_origin)

                if self.rotation:
                    angle = rand_rotation
                    if self.rotation_steps:
                        angle = utils.round_threshold(angle, self.rotation_steps)
                        # clamp angle in self.rotation
                        if angle > self.rotation:
                            angle -= self.rotation_steps
                        elif angle < -self.rotation:
                            angle += self.rotation_steps

                    if island.rotate(angle, vec_origin, self.aspect):
                        bb.rotate_expand(angle, self.aspect)

                scale = bl_math.lerp(1.0, rand_scale, self.scale_factor)

                # We choose a scale of 100 units on purpose, in case it will not be changed,
                # then the conditional check will not pass.
                new_scale = 100
                # Reset the scale to 0.5 to fit in the tile.
                if self.bool_bounds:
                    max_length = bb.max_length
                    max_length_lock = 1.0
                    if max_length * scale > max_length_lock:
                        new_scale = max_length_lock / max_length

                if self.scale_factor != 0 or new_scale < 1:
                    # If the scale from random is smaller, we choose it
                    scale = min(scale, new_scale)
                    scale = Vector((scale, scale))
                    island.scale(scale, pivot=vec_origin)
                    bb.scale(scale)

            move = Vector((self.strength.x, self.strength.y))

            if not (move.x or move.y):
                continue

            random.seed(seed + 2000)
            rand_move_x = 2 * (random.random() - 0.5)
            random.seed(seed + 3000)
            rand_move_y = 2 * (random.random() - 0.5)

            randmove = Vector((rand_move_x, rand_move_y)) * move

            if self.round_mode == 'INT':
                randmove = Vector([round(i) for i in randmove])
            elif self.round_mode == 'STEPS':
                # TODO bool_bounds for steps
                if self.steps.x > 1e-05:
                    randmove.x = utils.round_threshold(randmove.x, self.steps.x)
                if self.steps.y > 1e-05:
                    randmove.y = utils.round_threshold(randmove.y, self.steps.y)

            if not self.bool_bounds:
                island.move(randmove)
            else:
                min_bb_prev = island.bbox.tile_from_center
                bb = island.calc_bbox()
                wrap_x = utils.wrap_line(bb.xmin+randmove.x, bb.width, min_bb_prev.x, min_bb_prev.x+1, default=bb.min.x)
                wrap_y = utils.wrap_line(bb.ymin+randmove.y, bb.height, min_bb_prev.y, min_bb_prev.y+1, default=bb.min.y)

                # TODO: Crop island if max_length > 1.0
                delta = Vector((wrap_x, wrap_y))
                island.set_position(delta, bb.min)

    def random_between(self):
        shaffle_island_bboxes: list[BBox] = [isl.bbox.copy() for isl in self.all_islands]
        random.seed(self.seed)
        random.shuffle(shaffle_island_bboxes)


        for e_seed, island in enumerate(self.all_islands, start=100):
            seed = e_seed + self.seed
            random.seed(seed)
            rand_rotation = random.uniform(-self.rotation, self.rotation)
            random.seed(seed + 1000)
            rand_scale = random.uniform(self.min_scale, self.max_scale)

            flip_x = random.choices([-1, 1], weights=[self.flip_strength[0], 1 - self.flip_strength[0]], k=1)[0]
            flip_y = random.choices([-1, 1], weights=[self.flip_strength[1], 1 - self.flip_strength[1]], k=1)[0]

            bb = island.bbox
            if -1 in (flip_x, flip_y):
                island.scale(Vector((flip_x, flip_y)), pivot=bb.center)

            if self.rotation or self.scale_factor != 0:
                if bb.min_length == 0:
                    self.non_valid_counter += 1
                    continue

                vec_origin = bb.center

                if self.rotation:
                    angle = rand_rotation
                    if self.rotation_steps:
                        angle = utils.round_threshold(angle, self.rotation_steps)
                        # clamp angle in self.rotation
                        if angle > self.rotation:
                            angle -= self.rotation_steps
                        elif angle < -self.rotation:
                            angle += self.rotation_steps

                    island.rotate(angle, vec_origin, self.aspect)
                    # if island.rotate(angle, vec_origin):
                    #     bb.rotate_expand(angle)
                if self.bound_between != 'CROP':
                    scale = bl_math.lerp(1.0, rand_scale, self.scale_factor)
                    vector_scale = Vector((scale, scale))
                    island.scale(vector_scale, vec_origin)
                    # bb.scale(vector_scale, vec_origin)

            protege = shaffle_island_bboxes[e_seed-100]
            island.set_position(protege.center, _from=island.bbox.center)

            if self.bound_between == 'OFF':
                continue

            if self.lock_overlap:
                bb = island.calc_bbox(force=True)
            else:
                bb = island.calc_bbox()

            if protege.min_length < 2e-07 or bb.min_length < 2e-07:
                self.non_valid_counter += 1
                continue

            if self.bound_between == 'CLAMP':
                if bb.width <= protege.width:
                    width_scale = 1
                else:
                    width_scale = protege.width / bb.width

                if bb.height <= protege.height:
                    height_scale = 1
                else:
                    height_scale = protege.height / bb.height
            else:  # 'CROP'
                width_scale = protege.width / bb.width
                height_scale = protege.height / bb.height

            crop_scale = min(width_scale, height_scale)
            island.scale(Vector((crop_scale, crop_scale)), bb.center)


class UNIV_OT_Orient(Operator, utils.OverlapHelper):
    bl_idname = 'uv.univ_orient'
    bl_label = 'Orient'
    bl_description = "Rotated to a minimal rectangle, either vertical or horizontal\n\n" \
                     "Default - Fit by Islands\n" \
                     "Alt - Orient by Edge\n" \
                     "Has [O] keymap"
    bl_options = {'REGISTER', 'UNDO'}

    edge_dir: EnumProperty(name='Direction', default='HORIZONTAL', items=(
        ('BOTH', 'Both', ''),
        ('HORIZONTAL', 'Horizontal', ''),
        ('VERTICAL', 'Vertical', ''),
    ))
    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True)

    def draw(self, context):
        layout = self.layout
        layout.row().prop(self, 'edge_dir', expand=True)
        layout.prop(self, 'use_correct_aspect')
        self.draw_overlap()

    def invoke(self, context, event):
        self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
        self.mouse_pos = None
        if event.value == 'PRESS':
            if context.area.ui_type == 'UV':
                self.mouse_pos = utils.get_mouse_pos(context, event)
            return self.execute(context)

        self.lock_overlap = event.shift
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aspect: float = 1.0
        # WARNING: Possible potential error when calling via bpy.ops.uv.univ_orient('DEFAULT') when an old value is used
        self.mouse_pos: Vector | None = None
        self.max_distance: float | None = None
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.aspect = utils.get_aspect_ratio() if self.use_correct_aspect else 1.0
        self.umeshes = types.UMeshes(report=self.report)
        self.umeshes.update_tag = False

        selected_edges = []
        selected_faces, visible = self.umeshes.filtered_by_selected_and_visible_uv_faces()
        self.umeshes = selected_faces if selected_faces else visible
        if not selected_faces:
            selected_edges, visible = visible.filtered_by_selected_and_visible_uv_edges()
            self.umeshes = selected_edges if selected_edges else visible

        if not self.umeshes:
            return self.umeshes.update()

        if self.lock_overlap:
            if selected_faces:
                self.orient_islands_with_selected_faces_overlap(extended=True)
            elif selected_edges:
                self.orient_islands_with_selected_edges_overlap()
            else:
                self.orient_islands_with_selected_faces_overlap(extended=False)
        else:
            if selected_faces:
                self.orient_islands_with_selected_faces()
            elif selected_edges:
                self.orient_islands_with_selected_edges()
            else:
                return self.orient_pick_or_visible()
        self.umeshes.update(info="All islands oriented")
        return {'FINISHED'}

    def orient_islands_with_selected_faces(self):
        for umesh in self.umeshes:
            for island in Islands.calc_extended_with_mark_seam(umesh):
                self.orient_island(island)

    def orient_islands_with_selected_edges(self):
        for umesh in self.umeshes:
            for island in Islands.calc_extended_any_edge_with_markseam(umesh):
                self.orient_edge(island)

    def orient_pick_or_visible(self):
        hit = types.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            for isl in Islands.calc_visible_with_mark_seam(umesh):
                if self.mouse_pos:
                    hit.find_nearest_island_by_crn(isl)
                else:
                    self.orient_island(isl)

        if self.mouse_pos:
            if hit:
                uv = hit.island.umesh.uv
                v1 = hit.crn[uv].uv
                v2 = hit.crn.link_loop_next[uv].uv
                self.orient_edge_ex(hit.island, v1, v2)

                length = (v1 - v2).length
                if length:
                    return self.umeshes.update(info="Island oriented")
                else:
                    hit.crn[uv].select = True
                    hit.crn[uv].select_edge = True
                    hit.crn.link_loop_next[uv].select = True
                    hit.crn.face.select = True
                    hit.island.umesh.update_tag = True
                    self.report({'WARNING'}, "Island has zero length edge")
                    return self.umeshes.update()
            return self.umeshes.update(info_type={'WARNING'}, info="Island not found within a given radius")

        return self.umeshes.update(info="All islands oriented")

    def orient_islands_with_selected_faces_overlap(self, extended):
        islands_of_mesh = []
        for umesh in self.umeshes:
            if islands := AdvIslands.calc_extended_or_visible_with_mark_seam(umesh, extended=extended):
                islands.calc_tris()
                islands.calc_flat_coords(save_triplet=True)
                islands_of_mesh.extend(islands)

        for overlapped_isl in self.calc_overlapped_island_groups(islands_of_mesh):
            self.orient_island(overlapped_isl)

    def orient_islands_with_selected_edges_overlap(self):
        islands_of_mesh = []
        for umesh in self.umeshes:
            if islands := AdvIslands.calc_extended_any_edge_with_markseam(umesh):
                islands.calc_tris()
                islands.calc_flat_coords(save_triplet=True)
                islands_of_mesh.extend(islands)

        for overlapped_isl in self.calc_overlapped_island_groups(islands_of_mesh):
            self.orient_edge(overlapped_isl)

    def orient_edge(self, island):
        iter_isl = island if isinstance(island, types.UnionIslands) else (island,)
        max_length = -1.0
        v1 = Vector()
        v2 = Vector()
        for isl in iter_isl:
            uv = isl.umesh.uv
            for crn in isl.calc_selected_edge_corners_iter():
                v1_ = crn[uv].uv
                v2_ = crn.link_loop_next[uv].uv
                if (new_length := (v1_ - v2_).length) > max_length:
                    v1 = v1_
                    v2 = v2_
                    max_length = new_length

        if max_length != -1.0:
            self.orient_edge_ex(island, v1, v2)

    def orient_edge_ex(self, island, v1: Vector, v2: Vector):
        edge_vec: Vector = (v2 - v1) * Vector((self.aspect, 1.0))
        edge_vec.normalize()

        if not any(edge_vec):  # TODO: Use inspect (Zero)
            return

        if self.edge_dir == 'BOTH':
            current_angle = atan2(*edge_vec)
            angle_to_rotate = -utils.find_min_rotate_angle(current_angle)

        elif self.edge_dir == 'HORIZONTAL':
            left_dir = Vector((-1, 0))
            right_dir = Vector((1, 0))
            a = edge_vec.angle_signed(left_dir)
            b = edge_vec.angle_signed(right_dir)
            angle_to_rotate = a if abs(a) < abs(b) else b

        else:  # VERTICAL
            bottom_dir = Vector((0, -1))
            upper_dir = Vector((0, 1))
            a = edge_vec.angle_signed(bottom_dir)
            b = edge_vec.angle_signed(upper_dir)
            angle_to_rotate = a if abs(a) < abs(b) else b

        pivot: Vector = (v1 + v2) / 2
        island.umesh.update_tag |= island.rotate(angle_to_rotate, pivot, self.aspect)

    def orient_island(self, island: AdvIsland | types.UnionIslands):
        from collections import Counter
        angles: Counter[float | float] = Counter()
        boundary_coords = []
        is_boundary = utils.is_boundary_sync if island.umesh.sync else utils.is_boundary_non_sync

        iter_isl = island if isinstance(island, types.UnionIslands) else (island, )
        for isl_ in iter_isl:
            uv = isl_.umesh.uv
            vec_aspect = Vector((self.aspect, 1.0))

            boundary_corners = (crn for f in isl_ for crn in f.loops if crn.edge.seam or is_boundary(crn, uv))
            for crn in boundary_corners:
                v1 = crn[uv].uv
                v2 = crn.link_loop_next[uv].uv
                boundary_coords.append(v1)

                edge_vec: Vector = (v2 - v1) * vec_aspect
                if any(edge_vec):
                    current_angle = atan2(*edge_vec)
                    angle_to_rotate = -utils.find_min_rotate_angle(round(current_angle, 4))
                    angles[round(angle_to_rotate, 4)] += edge_vec.length

        if not angles:
            return
        # TODO: Calculate by convex if the angles are many (organic) and have ~ simular distances
        angle = max(angles, key=angles.get)

        bbox = types.BBox.calc_bbox(boundary_coords)
        island.umesh.update_tag |= island.rotate(angle, bbox.center, self.aspect)

        bbox = types.BBox.calc_bbox(boundary_coords)
        if self.edge_dir == 'HORIZONTAL':
            if bbox.width*self.aspect < bbox.height:
                final_angle = pi/2 if angle < 0 else -pi/2
                island.umesh.update_tag |= island.rotate(final_angle, bbox.center, self.aspect)

        elif self.edge_dir == 'VERTICAL':
            if bbox.width*self.aspect > bbox.height:
                final_angle = pi/2 if angle < 0 else -pi/2
                island.umesh.update_tag |= island.rotate(final_angle, bbox.center, self.aspect)


# The code was taken and modified from the TexTools addon: https://github.com/Oxicid/TexTools-Blender/blob/master/op_island_align_world.py
class UNIV_OT_Gravity(Operator):
    bl_idname = 'mesh.univ_gravity'
    bl_label = 'Gravity'
    bl_description = "Align selected UV islands or faces to world / gravity directions"
    bl_options = {'REGISTER', 'UNDO'}

    # axis: bpy.props.EnumProperty(name="Axis", default='AUTO', items=(
    #                                 ('AUTO', 'Auto', 'Detect World axis to align to.'),
    #                                 ('U', 'X', 'Align to the X axis of the World.'),
    #                                 ('V', 'Y', 'Align to the Y axis of the World.'),
    #                                 ('W', 'Z', 'Align to the Z axis of the World.')))
    additional_angle: FloatProperty(name='Additional Angle', default=0.0, soft_min=-pi/2, soft_max=pi, subtype='ANGLE')
    use_correct_aspect: BoolProperty(name='Correct Aspect', default=True, description='Gets Aspect Correct from the active image from the shader node editor')

    def draw(self, context):
        self.layout.prop(self, 'additional_angle', slider=True)
        self.layout.prop(self, 'use_correct_aspect', toggle=1)
        self.layout.row().prop(self, 'axis', expand=True)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = 'AUTO'
        self.skip_count: int = 0
        self.is_edit_mode: bool = bpy.context.mode == 'EDIT_MESH'
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
        self.umeshes.set_sync(True)

        if not self.is_edit_mode:
            self.umeshes.ensure(face=True)
            self.world_orient(extended=False)
            if self.skip_count == len(self.umeshes):
                self.umeshes.free()
                return self.umeshes.update(info="Faces not found")
        else:
            self.world_orient(extended=True)
            if self.skip_count == len(self.umeshes):
                self.world_orient(extended=False)
                if self.skip_count == len(self.umeshes):
                    return self.umeshes.update(info="No uv for manipulate")

        self.umeshes.update(info="All islands oriented")

        if not self.is_edit_mode:
            self.umeshes.free()
            bpy.context.area.tag_redraw()

        return {'FINISHED'}

    def world_orient(self, extended):
        self.skip_count = 0
        for umesh in self.umeshes:
            aspect = utils.get_aspect_ratio(umesh) if self.use_correct_aspect else 1.0
            umesh.update_tag = False
            if self.is_edit_mode:
                islands = Islands.calc_extended_or_visible(umesh, extended=extended)
            else:
                islands = Islands.calc_with_hidden(umesh)

            if islands:
                uv = islands.umesh.uv
                _, r, _ = umesh.obj.matrix_world.decompose()
                mtx = r.to_matrix()

                for island in islands:
                    if extended:
                        pre_calc_faces = (f for f in island if f.select)
                    else:
                        pre_calc_faces = island

                        # Get average viewport normal of UV island
                    calc_loops = []
                    avg_normal = Vector()
                    for face in pre_calc_faces:
                        avg_normal += face.normal * face.calc_area()
                        calc_loops.extend(face.loops)
                    avg_normal.rotate(mtx)

                    # Which Side
                    x, y, z = 0, 1, 2
                    max_size = max(avg_normal, key=lambda v: abs(v))

                    if (self.axis == 'AUTO' and avg_normal.z == max_size) or self.axis == 'W':
                        angle = self.calc_world_orient_angle(uv, mtx, calc_loops, x, y, False, avg_normal.z < 0, aspect)
                    elif (self.axis == 'AUTO' and avg_normal.y == max_size) or self.axis == 'V':
                        angle = self.calc_world_orient_angle(uv, mtx, calc_loops, x, z, avg_normal.y > 0, False, aspect)
                    else:  # (self.axis == 'AUTO' and avg_normal.x == max_size) or self.axis == 'U':
                        angle = self.calc_world_orient_angle(uv, mtx, calc_loops, y, z, avg_normal.x < 0, False, aspect)

                    if angle := (angle + self.additional_angle):
                        umesh.update_tag |= island.rotate(angle, pivot=island.calc_bbox().center, aspect=aspect)
            else:
                self.skip_count += 1

    @staticmethod
    def calc_world_orient_angle(uv, mtx, loops, x=0, y=1, flip_x=False, flip_y=False, aspect=1.0):
        vec_aspect = Vector((aspect, 1.0))

        n_edges = 0
        avg_angle = 0
        for loop in loops:
            co0 = mtx @ loop.vert.co
            co1 = mtx @ loop.link_loop_next.vert.co

            delta = co1 - co0
            max_side = max(map(abs, delta))

            # Check edges dominant in active axis
            if abs(delta[x]) == max_side or abs(delta[y]) == max_side:
                n_edges += 1
                uv0 = loop[uv].uv
                uv1 = loop.link_loop_next[uv].uv

                delta_verts = Vector((0, 0))
                if not flip_x:
                    delta_verts.x = co1[x] - co0[x]
                else:
                    delta_verts.x = co0[x] - co1[x]
                if not flip_y:
                    delta_verts.y = co1[y] - co0[y]
                else:
                    delta_verts.y = co0[y] - co1[y]

                delta_uvs = uv1 - uv0
                delta_uvs *= vec_aspect

                a0 = atan2(*delta_verts)
                a1 = atan2(*delta_uvs)

                a_delta = atan2(sin(a0 - a1), cos(a0 - a1))

                # Consolidation (atan2 gives the lower angle between -Pi and Pi,
                # this triggers errors when using the average avg_angle /= n_edges for rotation angles close to Pi)
                if n_edges > 1:
                    if abs((avg_angle / (n_edges - 1)) - a_delta) > 3.12:
                        if a_delta > 0:
                            avg_angle += (a_delta - pi * 2)
                        else:
                            avg_angle += (a_delta + pi * 2)
                    else:
                        avg_angle += a_delta
                else:
                    avg_angle += a_delta

        return avg_angle / n_edges


class UNIV_OT_Pack(Operator):
    bl_idname = 'uv.univ_pack'
    bl_label = 'Pack'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Pack selected islands\n\n" \
                     f"Has [P] keymap, but it conflicts with the 'Pin' operator"

    # def invoke(self, context, event):
    #     return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        if univ_settings().use_uvpm:
            if hasattr(context.scene, 'uvpm3_props'):
                return self.pack_uvpm()
            else:
                univ_settings().use_uvpm = False
                self.report({'WARNING'}, 'UVPackmaster not found')
                return {'CANCELLED'}
        else:
            return self.pack_native()

    @staticmethod
    def pack_uvpm():
        # TODO: Add Info about unselected and hidden faces
        # TODO: Use UniV orient (instead Pre-Rotation) and remove AXIS_ALIGNED method
        # TODO: Use UniV normalize
        # TODO: Add for Exact Overlap Mode threshold
        # TODO: Add scale checker for packed meshes

        settings = univ_settings()
        uvpm_settings = bpy.context.scene.uvpm3_props

        if hasattr(uvpm_settings, 'scale_mode'):
            uvpm_settings.scale_mode = '0' if settings.scale else '1'
        else:
            uvpm_settings.fixed_scale = not settings.scale

        uvpm_settings.pixel_margin_enable = True
        uvpm_settings.pixel_margin_tex_size = min(int(settings.size_x), int(settings.size_y))
        uvpm_settings.pixel_margin = settings.padding

        uvpm_settings.heuristic_enable = True
        total_selected = sum(umesh.total_face_sel for umesh in UMeshes.calc(verify_uv=False))
        if total_selected > 50_000:
            time_in_sec = 8
        elif total_selected > 10_000:
            time_in_sec = 4
        else:
            time_in_sec = 2
        uvpm_settings.heuristic_max_wait_time = time_in_sec
        uvpm_settings.heuristic_search_time = time_in_sec * 4

        if size := utils.get_active_image_size():
            uvpm_settings.tex_ratio = size[0] != size[1]
        else:
            uvpm_settings.tex_ratio = False

        if hasattr(uvpm_settings, 'default_main_props'):
            if uvpm_settings.default_main_props.precision == 500:
                uvpm_settings.default_main_props.precision = 800
        else:
            if uvpm_settings.precision == 500:
                uvpm_settings.precision = 800

        return bpy.ops.uvpackmaster3.pack('INVOKE_REGION_WIN', mode_id="pack.single_tile", pack_op_type='0')

    def pack_native(self):
        umeshes = UMeshes.calc(verify_uv=False)
        umeshes.fix_context()

        settings = univ_settings()
        args = {
            'udim_source': settings.udim_source,
            'rotate': settings.rotate,
            'margin': settings.padding / 2 / min(int(settings.size_x), int(settings.size_y))}
        if bpy.app.version >= (3, 5, 0):
            args['margin_method'] = 'FRACTION'
        is_360v = bpy.app.version >= (3, 6, 0)
        if is_360v:
            args['scale'] = settings.scale
            args['rotate_method'] = settings.rotate_method
            args['pin'] = settings.pin
            args['merge_overlap'] = settings.merge_overlap
            args['pin_method'] = settings.pin_method
            args['shape_method'] = settings.shape_method

        import platform
        if is_360v and settings.shape_method != 'AABB' and platform.system() == 'Windows':
            import threading
            threading.Thread(target=self.press_enter_key).start()
            return bpy.ops.uv.pack_islands('INVOKE_DEFAULT', **args)  # noqa
        else:
            return bpy.ops.uv.pack_islands('EXEC_DEFAULT', **args)  # noqa

    @staticmethod
    def press_enter_key():
        import ctypes
        VK_RETURN = 0x0D  # Enter  # noqa
        KEYDOWN = 0x0000  # Press  # noqa
        KEYUP = 0x0002  # Release  # noqa
        ctypes.windll.user32.keybd_event(VK_RETURN, 0, KEYDOWN, 0)
        ctypes.windll.user32.keybd_event(VK_RETURN, 0, KEYUP, 0)
