# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import random
import bl_math
from collections.abc import Callable

import numpy as np

from bpy.types import Operator
from bpy.props import *

from math import pi, sin, cos, atan2, sqrt, isclose
from mathutils import Vector, Matrix
from mathutils.geometry import area_tri
from collections import defaultdict

from bmesh.types import BMLoop

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
    UnionIslands,
    LoopGroup
)
from ..preferences import prefs, univ_settings


class UNIV_OT_Crop(Operator):
    bl_idname = 'uv.univ_crop'
    bl_label = 'Crop'
    bl_description = info.operator.crop_info
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(name='Axis', default='XY', items=(('XY', 'Both', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    to_cursor: BoolProperty(name='To Cursor', default=False)
    individual: BoolProperty(name='Individual', default=False)
    inplace: BoolProperty(name='Inplace', default=False)
    padding: FloatProperty(name='Padding', description='Padding=1/TextureSize (1/256=0.0039)', default=0, soft_min=0, soft_max=1/256*4, max=0.49)

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
        layout.prop(self, 'padding', slider=True)

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
        delta = Vector((padding, padding)) / 2 - bbox.min + offset
        if axis == 'XY':
            if inplace:
                delta += Vector(math.floor(val) for val in bbox.center)
        elif axis == 'X':
            delta.y = 0
            if inplace:
                delta.x += math.floor(bbox.center_x)
        else:
            delta.x = 0
            if inplace:
                delta.y += math.floor(bbox.center_y)
        for islands in islands_of_mesh:
            islands.scale(scale, bbox.center)
            islands.move(delta)

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
        return self.crop(self.mode, self.axis, self.padding, proportional=False, report=self.report)

    @staticmethod
    def get_event_info():
        return info.operator.fill_event_info_ex


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
class UNIV_OT_Align(Operator):
    bl_idname = 'uv.univ_align'
    bl_label = 'Align'
    bl_description = info.operator.align_info
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name="Mode", default='ALIGN', items=(
        ('ALIGN', 'Align', ''),
        ('INDIVIDUAL_OR_MOVE', 'Individual | Move', ''),
        ('ALIGN_CURSOR', 'Move cursor to selected', ''),
        ('ALIGN_TO_CURSOR', 'Align to cursor', ''),
        ('ALIGN_TO_CURSOR_UNION', 'Align to cursor union', ''),
        ('CURSOR_TO_TILE', 'Align cursor to tile', ''),
        # ('MOVE_COLLISION', 'Collision move', '')
    ))

    direction: EnumProperty(name="Direction", default='UPPER', items=align_align_direction_items)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(self, 'direction')
        self.layout.column(align=True).prop(self, 'mode', expand=True)

    def invoke(self, context, event):
        match event.ctrl, event.shift, event.alt:
            case False, False, False:
                self.mode = 'ALIGN'
            case True, False, False:
                self.mode = 'ALIGN_TO_CURSOR'
            case True, True, True:
                self.mode = 'ALIGN_TO_CURSOR_UNION'
            case False, False, True:
                self.mode = 'ALIGN_CURSOR'
            case True, False, True:
                self.mode = 'CURSOR_TO_TILE'
            case False, True, False:
                self.mode = 'INDIVIDUAL_OR_MOVE'
            case _:
                self.report({'INFO'}, f"Event: {utils.event_to_string(event)} not implement. \n\n"
                                      f"See all variations:\n\n{info.operator.align_event_info_ex}")
                return {'CANCELLED'}
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes = None

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
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

            case 'CURSOR_TO_TILE':
                if not (cursor_loc := utils.get_cursor_location()):
                    self.umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                self.align_cursor_to_tile(cursor_loc)
                return {'FINISHED'}

            case 'INDIVIDUAL_OR_MOVE':  # OR INDIVIDUAL
                if not utils.is_island_mode():
                    self.individual_scale_zero()
                else:
                    self.move_ex(selected=True)
                if not self.umeshes.final():
                    if self.direction in {'CENTER', 'HORIZONTAL', 'VERTICAL'}:
                        self.align_ex(selected=False)
                    else:
                        self.move_ex(selected=False)

            case _:
                raise NotImplementedError(self.mode)

        return self.umeshes.update()

    def move_to_cursor_ex(self, cursor_loc, selected=True):
        all_groups = []  # islands, bboxes, uv or corners, uv
        general_bbox = BBox.init_from_minmax(cursor_loc, cursor_loc)
        if utils.is_island_mode() or (not selected and self.direction not in {'LEFT', 'RIGHT', 'BOTTOM', 'UPPER'}):
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
        all_groups = []  # islands, bboxes, uv or corners, uv
        general_bbox = BBox()
        for umesh in self.umeshes:
            if corners := utils.calc_uv_corners(umesh, selected=selected):  # TODO: Implement bbox by individual modes
                all_groups.append((corners, umesh.uv))
                bbox = BBox.calc_bbox_uv_corners(corners, umesh.uv)
                general_bbox.union(bbox)
        return general_bbox

    def align_ex(self, selected=True):
        all_groups = []  # islands, bboxes, uv or corners, uv
        general_bbox = BBox()
        if utils.is_island_mode() or not selected:
            for umesh in self.umeshes:
                if islands := Islands.calc_extended_or_visible(umesh, extended=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        general_bbox.union(bbox)

                        all_groups.append((island, bbox, umesh.uv))
                umesh.update_tag = bool(islands)
            self.align_islands(all_groups, general_bbox)
        else:
            for umesh in self.umeshes:
                if corners := utils.calc_uv_corners(umesh, selected=selected):
                    bbox = BBox.calc_bbox_uv_corners(corners, umesh.uv)
                    general_bbox.union(bbox)

                    all_groups.append((corners, umesh.uv))
                umesh.update_tag = bool(corners)
            self.align_corners(all_groups, general_bbox)  # TODO Individual ALign for Vertical and Horizontal or all

    def move_ex(self, selected=True):
        if utils.is_island_mode():
            for umesh in self.umeshes:
                if islands := Islands.calc_extended_or_visible(umesh, extended=selected):
                    match self.direction:
                        case 'CENTER':
                            for island in islands:
                                bbox = island.calc_bbox()
                                delta = Vector((0.5, 0.5)) - bbox.center
                                island.move(delta)  # ToDO: Implement update flags for return state move
                        case 'HORIZONTAL':
                            for island in islands:
                                bbox = island.calc_bbox()
                                delta_y = 0.5 - bbox.center.y
                                island.move(Vector((0.0, delta_y)))
                        case 'VERTICAL':
                            for island in islands:
                                bbox = island.calc_bbox()
                                delta_x = 0.5 - bbox.center.x
                                island.move(Vector((delta_x, 0.0)))
                        case _:
                            move_value = Vector(self.get_move_value(self.direction))
                            for island in islands:
                                island.move(move_value)
                umesh.update_tag = bool(islands)
        else:
            for umesh in self.umeshes:
                if corners := utils.calc_uv_corners(umesh, selected=selected):
                    uv = umesh.uv
                    match self.direction:
                        case 'CENTER':
                            for corner in corners:
                                corner[uv].uv = 0.5, 0.5
                        case 'HORIZONTAL':
                            for corner in corners:
                                corner[uv].uv.x = 0.5
                        case 'VERTICAL':
                            for corner in corners:
                                corner[uv].uv.y = 0.5
                        case _:
                            move_value = Vector(self.get_move_value(self.direction))
                            for corner in corners:
                                corner[uv].uv += move_value
                umesh.update_tag = bool(corners)

    def individual_scale_zero(self):
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

    def align_cursor_to_tile(self, cursor_loc):
        def pad_floor(value):
            f = np.floor(np.float16(value))
            return np.nextafter(f, f + np.float16(1.0))

        def pad_ceil(value):
            f = np.floor(np.float16(value))
            return np.nextafter(f + np.float16(1.0), f)

        x, y = cursor_loc
        match self.direction:
            case 'UPPER':
                y = pad_ceil(y)
            case 'BOTTOM':
                y = pad_floor(y)
            case 'LEFT':
                x = pad_floor(x)
            case 'RIGHT':
                x = pad_ceil(x)
            case 'RIGHT_UPPER':
                x = pad_ceil(x)
                y = pad_ceil(y)
            case 'LEFT_UPPER':
                x = pad_floor(x)
                y = pad_ceil(y)
            case 'LEFT_BOTTOM':
                x = pad_floor(x)
                y = pad_floor(y)
            case 'RIGHT_BOTTOM':
                x = pad_ceil(x)
                y = pad_floor(y)
            case 'CENTER':
                x = np.floor(x) + 0.5
                y = np.floor(y) + 0.5
            case 'HORIZONTAL':
                y = np.floor(y) + 0.5
            case 'VERTICAL':
                x = np.floor(x) + 0.5
            case _:
                raise NotImplementedError(self.direction)
        utils.set_cursor_location((x, y))

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


class UNIV_OT_Sort(Operator, utils.OverlapHelper):
    bl_idname = 'uv.univ_sort'
    bl_label = 'Sort'
    bl_description = 'Sort'
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(name='Axis', default='AUTO', items=(('AUTO', 'Auto', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    padding: FloatProperty(name='Padding', default=1/2048, min=0, soft_max=0.1,)
    sub_padding: FloatProperty(name='Sub Padding', default=0.1, min=0, soft_max=0.2,)
    area_subgroups: IntProperty(name='Area Subgroups', default=4, min=1, max=200, soft_max=8)
    reverse: BoolProperty(name='Reverse', default=True)
    to_cursor: BoolProperty(name='To Cursor', default=False)
    align: BoolProperty(name='Orient', default=False)  # TODO: Rename align to orient
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
        layout.prop(self, 'align')
        if self.subgroup_type == 'NONE':
            self.draw_overlap()
        layout.prop(self, 'padding', slider=True)
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
        self.align = event.alt
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
            if self.align:
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
                if self.align:
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
                if self.align and island.bbox.height < width:
                    width = island.bbox.height
                    self.update_tag |= island.rotate(pi * 0.5, island.bbox.center)
                    island.calc_bbox()
                self.update_tag |= island.set_position(margin, _from=island.bbox.min)
                margin.x += self.padding + width
            margin.x += self.sub_padding
        else:
            for island in islands:
                height = island.bbox.height
                if self.align and island.bbox.width < height:
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


class UNIV_OT_Distribute(Operator, utils.OverlapHelper):
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
    padding: FloatProperty(name='Padding', default=1/2048, min=0, soft_max=0.1,)
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

        layout = self.layout
        layout.prop(self, 'padding', slider=True)
        layout = self.layout.row()
        layout.prop(self, 'axis', expand=True)

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
                self.report({'WARNING'}, self.no_change_info)
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
            self.report({'WARNING'}, self.no_change_info)
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
        all_object = set(types.UMeshes.calc_all_objects(verify_uv=False)) - set(self.umeshes)
        mod_counter = 0
        attr_counter = 0
        for umesh in self.umeshes:
            for mod in reversed(umesh.obj.modifiers):
                if isinstance(mod, bpy.types.NodesModifier) and mod.name.startswith('UniV Shift'):
                    umesh.obj.modifiers.remove(mod)
                    mod_counter += 1

            # safe attr for instances if not zero
            instances = (inst_umesh for inst_umesh in all_object if inst_umesh.obj.data == umesh.obj.data)
            has_inst_with_shift_mod = False
            for i_mesh in instances:
                for mod in i_mesh.obj.modifiers:
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
            if any(other_umesh.obj.data == umesh.obj.data for umesh in self.umeshes):
                if self.umeshes.is_edit_mode:
                    attributes = other_umesh.bm.faces.layers.int.values()
                else:
                    attributes = other_umesh.obj.data.attributes

                if not any(attr.name.startswith('univ_shift') for attr in attributes):
                    for mod in reversed(other_umesh.obj.modifiers):
                        if isinstance(mod, bpy.types.NodesModifier) and mod.name.startswith('UniV Shift'):
                            other_umesh.obj.modifiers.remove(mod)
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

    between: BoolProperty(name='Between', default=False)
    bound_between: EnumProperty(name='Bound Between', default='OFF', items=(('OFF', 'Off', ''), ('CROP', 'Crop', ''), ('CLAMP', 'Clamp', '')))
    round_mode: EnumProperty(name='Round Mode', default='OFF', items=(('OFF', 'Off', ''), ('INT', 'Int', ''), ('STEPS', 'Steps', '')))
    steps: FloatVectorProperty(name='Steps', description="Incorrectly works with Within Image Bounds",
                               default=(0, 0), min=0, max=10, soft_min=0, soft_max=1, size=2, subtype='XYZ')
    strength: FloatVectorProperty(name='Strength', default=(1, 1), min=-10, max=10, soft_min=0, soft_max=1, size=2, subtype='XYZ')
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
        bb_general = BBox()
        for e_seed, island in enumerate(self.all_islands, start=100):
            seed = e_seed + self.seed
            random.seed(seed)
            rand_rotation = random.uniform(-self.rotation, self.rotation)
            random.seed(seed + 1000)
            rand_scale = random.uniform(self.min_scale, self.max_scale)

            if self.bool_bounds or self.rotation or self.scale_factor != 0:

                bb = island.bbox
                if bb.min_length == 0:
                    self.non_valid_counter += 1
                    continue

                vec_origin = bb.center

                if self.rotation:
                    angle = rand_rotation
                    if self.rotation_steps:
                        angle = self.round_threshold(angle, self.rotation_steps)
                        # clamp angle in self.rotation
                        if angle > self.rotation:
                            angle -= self.rotation_steps
                        elif angle < -self.rotation:
                            angle += self.rotation_steps

                    if island.rotate(angle, vec_origin, self.aspect):
                        bb.rotate_expand(angle, self.aspect)

                scale = bl_math.lerp(1.0, rand_scale, self.scale_factor)

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

                if self.bool_bounds:
                    to_center_delta = Vector((0.5, 0.5)) - vec_origin
                    island.move(to_center_delta)
                    bb.move(to_center_delta)
                    bb_general.union(bb)

            if self.bool_bounds:
                move = Vector((
                    min(bb_general.xmin, abs(1 - bb_general.xmax)) * max(min(self.strength.x, 1), -1),
                    min(bb_general.ymin, abs(1 - bb_general.ymax)) * max(min(self.strength.y, 1), -1)
                ))
            else:
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
                    randmove.x = self.round_threshold(randmove.x, self.steps.x)
                if self.steps.y > 1e-05:
                    randmove.y = self.round_threshold(randmove.y, self.steps.y)

            if not self.bool_bounds:
                island.move(randmove)

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

            bb = island.bbox
            if self.rotation or self.scale_factor != 0:
                if bb.min_length == 0:
                    self.non_valid_counter += 1
                    continue

                vec_origin = bb.center

                if self.rotation:
                    angle = rand_rotation
                    if self.rotation_steps:
                        angle = self.round_threshold(angle, self.rotation_steps)
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

    @staticmethod
    def round_threshold(a, min_clip):
        return round(float(a) / min_clip) * min_clip


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

        for umesh in self.umeshes:
            umesh.update_tag = False

        if self.lock_overlap:
            return self.orient_lock_overlap_processing()
        else:
            return self.orient_processing()

    def orient_processing(self):
        islands_of_mesh = []
        has_any_selected_elements = False  # This check is necessary because the selected edge can be without faces
        selected, visible = self.umeshes.filtered_by_selected_and_visible_uv_edges()

        for umesh in selected:
            if islands := Islands.calc_visible_with_mark_seam(umesh):
                islands_of_mesh.append(islands)
                if self.umeshes.sync and umesh.is_full_edge_deselected:
                    continue
                for island in islands:
                    has_selected_edges, has_selected_faces = self.has_selected_edges_or_faces(island)

                    if has_selected_faces:
                        self.orient_island(island)
                    elif has_selected_edges:
                        self.orient_edge(island)
                    has_any_selected_elements |= has_selected_edges or has_selected_faces

        if not has_any_selected_elements:
            for umesh in visible:
                if islands := Islands.calc_visible_with_mark_seam(umesh):
                    islands_of_mesh.append(islands)

            hit = types.IslandHit(self.mouse_pos, self.max_distance)
            for islands in islands_of_mesh:
                for isl in islands:
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
                return self.umeshes.update(info_type={'WARNING'}, info="Island not found")

        if not islands_of_mesh:
            self.report({'WARNING'}, "Islands not found")
            return {"CANCELLED"}

        return self.umeshes.update(info="All islands oriented")

    def orient_lock_overlap_processing(self):
        islands_with_selected_edges = []
        islands_with_selected_faces = []
        unselected_islands = []

        for umesh in self.umeshes:
            if islands := AdvIslands.calc_visible_with_mark_seam(umesh):
                islands.calc_tris()
                islands.calc_flat_coords(save_triplet=True)
                for island in islands:
                    has_selected_edges, has_selected_faces = self.has_selected_edges_or_faces(island)

                    if has_selected_edges:
                        islands_with_selected_edges.append(island)
                    elif has_selected_faces:
                        islands_with_selected_faces.append(island)
                    else:
                        unselected_islands.append(island)

        if not any((islands_with_selected_faces, islands_with_selected_edges, unselected_islands)):
            self.report({'WARNING'}, "Islands not found")
            return {"CANCELLED"}

        if islands_with_selected_edges:
            for overlapped_isl in self.calc_overlapped_island_groups(islands_with_selected_edges):
                self.orient_edge(overlapped_isl)

        if islands_with_selected_faces:
            for overlapped_isl in self.calc_overlapped_island_groups(islands_with_selected_faces):
                self.orient_island(overlapped_isl)

        if not any((islands_with_selected_edges, islands_with_selected_faces)):
            for overlapped_isl in self.calc_overlapped_island_groups(unselected_islands):
                self.orient_island(overlapped_isl)

        return self.umeshes.update(info="All islands oriented")

    @staticmethod
    def has_selected_edges_or_faces(island):
        has_selected_edges = False
        has_selected_faces = False

        if island.umesh.sync:
            if island.umesh.total_face_sel and any(f.select for f in island):  # orient island
                has_selected_faces = True
            elif island.umesh.total_edge_sel and any(e.select for f in island for e in f.edges):  # orient by edges
                has_selected_edges = True
        else:
            if island.umesh.total_face_sel:
                uv = island.umesh.uv
                for f in island:
                    if has_selected_faces:
                        break
                    corners = f.loops
                    counter = 0
                    for crn in corners:
                        select_state = crn[uv].select_edge
                        counter += select_state
                        has_selected_edges |= select_state
                    has_selected_faces = len(corners) == counter  # optimized, instead any
        return has_selected_edges, has_selected_faces

    def orient_edge(self, island):
        iter_isl = island if isinstance(island, types.UnionIslands) else (island,)
        max_length = -1.0
        v1 = Vector()
        v2 = Vector()
        for isl_ in iter_isl:
            uv = isl_.umesh.uv
            if isl_.umesh.sync:
                corners = (crn for f in isl_ for crn in f.loops if crn.edge.select)
            else:
                corners = (crn for f in isl_ for crn in f.loops if crn[uv].select_edge)
            for crn_ in corners:
                v1_ = crn_[uv].uv
                v2_ = crn_.link_loop_next[uv].uv
                if (new_length := (v1_ - v2_).length) > max_length:
                    v1 = v1_
                    v2 = v2_
                    max_length = new_length

        if max_length == 1.0:
            return
        self.orient_edge_ex(island, v1, v2)

    def orient_edge_ex(self, island, v1: Vector, v2: Vector):
        diff: Vector = (v2 - v1) * Vector((self.aspect, 1.0))

        if not any(diff):  # TODO: Use inspect (Zero)
            return
        diff.normalize()
        if self.edge_dir == 'BOTH':
            current_angle = atan2(*diff)
            angle_to_rotate = -utils.find_min_rotate_angle(current_angle)
        elif self.edge_dir == 'HORIZONTAL':
            vec = diff.normalized()
            angle_to_rotate = a if abs(a := vec.angle_signed(Vector((-1, 0)))) < abs(b := vec.angle_signed(Vector((1, 0)))) else b
        else:
            vec = diff.normalized()
            angle_to_rotate = a if abs(a := vec.angle_signed(Vector((0, -1)))) < abs(b := vec.angle_signed(Vector((0, 1)))) else b

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
            boundary_corners = (crn for f in isl_ for crn in f.loops if crn.edge.seam or is_boundary(crn, uv))

            vec_aspect = Vector((self.aspect, 1.0))
            for crn in boundary_corners:
                v1 = crn[uv].uv
                v2 = crn.link_loop_next[uv].uv
                boundary_coords.append(v1)

                diff: Vector = (v2 - v1) * vec_aspect

                if not any(diff):
                    continue

                current_angle = atan2(*diff)
                angle_to_rotate = -utils.find_min_rotate_angle(round(current_angle, 4))
                angles[round(angle_to_rotate, 4)] += diff.length

        if not angles:
            return

        # TODO: Calculate by convex if the angles are many and have ~ simular distances
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
        self.umeshes: types.UMeshes | None = None
        self.global_counter = 0
        self.seam_clear_counter = 0
        self.edge_weld_counter = 0
        self.max_distance: float = 0.0
        self.mouse_position: Vector | None = None
        self.stitched_islands = 0
        self.update_seams = True

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
        self.global_counter = 0
        self.seam_clear_counter = 0
        self.edge_weld_counter = 0
        self.stitched_islands = 0

        if self.use_by_distance:
            if self.weld_by_distance_type == 'BY_ISLANDS':
                self.weld_by_distance_island(extended=True)
                if not self.umeshes.final():
                    self.weld_by_distance_island(extended=False)
            else:
                self.weld_by_distance_all(selected=True)
                if not self.umeshes.final():
                    self.weld_by_distance_all(selected=False)
        else:
            self.weld()

            if self.stitched_islands:
                self.report({'INFO'}, f"Stitched {self.stitched_islands} ")
            else:
                if self.edge_weld_counter and self.seam_clear_counter:
                    self.report({'INFO'}, f"Welded {self.edge_weld_counter} edges. Cleared mark seams edges = {self.seam_clear_counter} ")
                elif self.edge_weld_counter:
                    self.report({'INFO'}, f"Welded {self.edge_weld_counter} edges.")
                elif self.seam_clear_counter:
                    self.report({'INFO'}, f"Cleared seams edges = {self.seam_clear_counter} ")

        self.umeshes.update(info='Not found verts for weld')
        return {'FINISHED'}

    def weld(self):
        from ..utils import weld_crn_edge_by_idx

        selected_umeshes, visible_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_edges()
        self.umeshes = selected_umeshes if selected_umeshes else visible_umeshes

        if not self.umeshes:
            return self.umeshes.update()
        if not selected_umeshes and self.mouse_position:
            self.umeshes.umeshes = []
            return
            # return self.pick_weld()


        islands_of_mesh = []
        for umesh in self.umeshes:
            uv = umesh.uv

            local_seam_clear_counter = 0
            local_edge_weld_counter = 0

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
                            local_edge_weld_counter += 1
                        elif is_splitted_a:
                            weld_crn_edge_by_idx(crn, shared_next, idx, uv)
                            local_edge_weld_counter += 1
                        elif is_splitted_b:
                            weld_crn_edge_by_idx(crn_next, shared, idx, uv)
                            local_edge_weld_counter += 1

                        edge = crn.edge
                        if edge.seam:
                            edge.seam = False
                            local_seam_clear_counter += 1

                        crn.tag = False
                        shared.tag = False

            self.seam_clear_counter += local_seam_clear_counter
            self.edge_weld_counter += local_edge_weld_counter
            umesh.update_tag = bool(local_seam_clear_counter + local_edge_weld_counter)

            if islands:
                islands_of_mesh.append(islands)

        if not self.umeshes or (self.seam_clear_counter + self.edge_weld_counter):
            return

        if not self.umeshes.sync:
            for islands in islands_of_mesh:
                uv = islands.umesh.uv

                local_seam_clear_counter = 0
                local_edge_weld_counter = 0

                for idx, isl in enumerate(islands):
                    for crn in isl.iter_corners_by_tag():
                        utils.copy_pos_to_target_with_select(crn, uv, idx)

                        edge = crn.edge
                        if crn.edge.seam:
                            edge.seam = False
                            local_seam_clear_counter += 1
                        local_edge_weld_counter += 1

                self.seam_clear_counter += local_seam_clear_counter
                self.edge_weld_counter += local_edge_weld_counter
                islands.umesh.update_tag = bool(local_seam_clear_counter + local_edge_weld_counter)

            if self.seam_clear_counter + self.edge_weld_counter:
                return

        UNIV_OT_Stitch.stitch(self)  # noqa TODO: Implement inheritance

    def weld_by_distance_island(self, extended):
        for umesh in self.umeshes:
            uv = umesh.uv
            local_counter = 0
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

                    corners = (crn for f in isl for crn in f.loops if crn.tag)
                    for crn in corners:
                        crn_in_vert = [crn_v for crn_v in crn.vert.link_loops if crn_v.tag]
                        local_counter += self.weld_corners_in_vert(crn_in_vert, uv)

            if local_counter:
                for _isl in islands:
                    _isl.mark_seam()
            umesh.update_tag = bool(islands)
            self.global_counter += local_counter

        if self.umeshes.final() and self.global_counter == 0:
            self.umeshes.cancel_with_report(info='Not found verts for weld')

        if self.global_counter:
            self.report({'INFO'}, f"Found {self.global_counter} vertices for weld")

    def weld_by_distance_all(self, selected):
        # TODO: Refactor this, use iterator
        for umesh in self.umeshes:
            umesh.tag_visible_corners()
            uv = umesh.uv
            local_counter = 0
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
                    local_counter += self.weld_corners_in_vert(crn_in_vert, uv)

            if local_counter:
                umesh.tag_visible_faces()
                umesh.mark_seam_tagged_faces()

            umesh.update_tag = bool(init_corners)
            self.global_counter += local_counter

        if self.umeshes.final() and self.global_counter == 0:
            self.umeshes.cancel_with_report(info='Not found verts for weld')

        if self.global_counter:
            self.report({'INFO'}, f"Found {self.global_counter} vertices for weld")

    def weld_corners_in_vert(self, crn_in_vert, uv):
        if utils.all_equal(_crn[uv].uv for _crn in crn_in_vert):
            for crn_t in crn_in_vert:
                crn_t.tag = False
            return 0
        sub_local_counter = 0
        for group in self.calc_distance_groups(crn_in_vert, uv):
            value = Vector((0, 0))
            for c in group:
                value += c[uv].uv
            size = len(group)
            avg = value / size
            for c in group:
                c[uv].uv = avg
            sub_local_counter += size
        return sub_local_counter

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
        else:
            shared = hit.crn.link_loop_radial_prev
            if (shared == hit.crn or
                    bool(shared.face.hide if hit.umesh.sync else not shared.face.select)):
                self.report({'WARNING'}, 'Edge is boundary')
                return

            if hit.crn.link_loop_next.vert != shared.vert:
                self.report({'WARNING'}, 'Edge has 3D flipped face')
                return

            e = hit.crn.edge
            had_seam = e.seam
            if not had_seam:
                hit.crn.edge.seam = True
                hit.umesh.update()

            return {'FINISHED'} if had_seam else {'FINISHED'}


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

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def invoke(self, context, event):
        if event.value == 'PRESS':
            if context.area.ui_type == 'UV':
                self.mouse_position = Vector(context.region.view2d.region_to_view(event.mouse_region_x, event.mouse_region_y))
            return self.execute(context)
        self.between = event.alt
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync = utils.sync()
        self.umeshes: types.UMeshes | None = None
        self.global_counter = 0
        self.mouse_position: Vector | None = None
        self.stitched_islands = 0
        self.save_seams = False

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
        self.global_counter = 0
        self.stitched_islands = 0
        if self.between:
            self.stitch_between()
        else:
            self.stitch()
            if self.stitched_islands:
                self.report({'INFO'}, f"Stitched {self.stitched_islands} ")

        return self.umeshes.update(info='Not found edges for stitch')

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
            self.stitched_islands += len(adv_islands) - sum(bool(isl) for isl in adv_islands)
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

    def stitch_ex(self, tar: LoopGroup, source: LoopGroup, adv_islands: AdvIslands, selected=True):
        uv = tar.umesh.uv
        # Equal indices occur after merging on non-stitch edges
        # TODO: Disable?
        if tar[0].face.index == source[0].face.index:
            for target_crn in tar:
                UNIV_OT_Stitch.copy_pos(target_crn, uv)
            return True

        if (corners := UNIV_OT_Stitch.calc_begin_end_points(tar, source)) is False:
            tar.tag = False
            return False

        target_isl = adv_islands[corners[0].face.index]
        source_isl = adv_islands[corners[2].face.index]

        # if not (target_isl.is_full_flipped != source_isl.is_full_flipped):  # TODO: Implement auto flip (reverse LoopGroup?)
        #     source_isl.scale_simple(Vector((1, -1)))
        if self.flip:
            source_isl.scale_simple(Vector((1, -1)))
        pt_a1, pt_a2, pt_b1, pt_b2 = [c[uv].uv for c in corners]

        normal_a = pt_a1 - pt_a2
        normal_b = pt_b1 - pt_b2

        # Scale
        scale = normal_a.length / normal_b.length
        source_isl.scale_simple(Vector((scale, scale)))

        # Rotate
        rotate_angle = normal_a.angle_signed(normal_b)
        source_isl.rotate_simple(rotate_angle)

        # Move
        source_isl.move(pt_a1 - pt_b1)

        adv_islands.weld_selected(target_isl, source_isl, selected=selected)
        return True


class UNIV_OT_ResetScale(Operator, utils.OverlapHelper):
    bl_idname = "uv.univ_reset_scale"
    bl_label = 'Reset'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Reset the scale of separate UV islands, based on their area in 3D space\n\n" \
                     f"Default - Reset islands scale\n" \
                     f"Shift - Lock Overlaps"

    shear: BoolProperty(name='Shear', default=True, description='Reduce shear within islands')
    axis: EnumProperty(name='Axis', default='XY', items=(('XY', 'Both', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    use_aspect: BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.lock_overlap = event.shift
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        self.draw_overlap()
        layout.row(align=True).prop(self, 'axis', expand=True)
        layout.prop(self, 'shear')
        layout.prop(self, 'use_aspect')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
        for umesh in self.umeshes:
            umesh.update_tag = False
            umesh.value = umesh.check_uniform_scale(report=self.report)

        all_islands: list[AdvIsland | UnionIslands] = []

        islands_calc_type: Callable[[types.UMesh], AdvIslands]
        if self.umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            self.umeshes = selected_umeshes if selected_umeshes else unselected_umeshes
            islands_calc_type = AdvIslands.calc_extended_with_mark_seam if selected_umeshes else AdvIslands.calc_visible_with_mark_seam
        else:
            islands_calc_type = AdvIslands.calc_with_hidden
            for umesh in self.umeshes:
                umesh.ensure(face=True)

        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0
            adv_islands = islands_calc_type(umesh)
            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'
            all_islands.extend(adv_islands)
            adv_islands.calc_tris_simple()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale

        if not all_islands:
            self.report({'WARNING'}, 'Islands not found')
            return {'CANCELLED'}

        if self.lock_overlap:
            all_islands = self.calc_overlapped_island_groups(all_islands)

        for isl in all_islands:
            isl.value = isl.bbox.center  # isl.value == pivot
            # TODO: Find how to calculate the shear for the X axis when aspect != 1 without rotation island
            if self.axis == 'X' and isl.umesh.aspect != 1.0 and self.shear:
                isl.rotate_simple(pi/2, isl.umesh.aspect)
                self.individual_scale(isl, 'Y',  self.shear)
                isl.rotate_simple(-pi/2, isl.umesh.aspect)
                new_center = isl.calc_bbox().center
            else:
                new_center = self.individual_scale(isl, self.axis, self.shear)
            isl.set_position(isl.value, new_center)

        self.umeshes.update(info='All islands were with scaled')

        if not self.umeshes.is_edit_mode:
            self.umeshes.free()
            utils.update_area_by_type('VIEW_3D')

        return {'FINISHED'}

    @staticmethod
    def individual_scale(isl: AdvIsland, axis, shear):
        from bl_math import clamp
        aspect = isl.umesh.aspect
        new_center = isl.value.copy()

        vectors_ac_bc = [(va - vc, vb - vc) for va, vb, vc in isl.flat_3d_coords]
        uv_coords_and_3d_vectors_and_3d_areas = tuple(zip(isl.flat_coords, vectors_ac_bc, isl.weights))
        for j in range(10):
            scale_cou = 0.0
            scale_cov = 0.0
            scale_cross = 0.0

            for (uv_a, uv_b, uv_c), (vec_ac, vec_bc), weight in uv_coords_and_3d_vectors_and_3d_areas:
                if isclose(area_tri(uv_a, uv_b, uv_c), 0, abs_tol=1e-9):
                    continue
                m = Matrix((uv_a - uv_c, uv_b - uv_c))
                try:
                    m.invert()
                except ValueError:
                    continue

                cou = m[0][0] * vec_ac + m[0][1] * vec_bc
                cov = m[1][0] * vec_ac + m[1][1] * vec_bc

                scale_cou += cou.length * weight
                scale_cov += cov.length * weight

                if shear:
                    cov.normalize()
                    cou.normalize()
                    scale_cross += cou.dot(cov) * weight

            if scale_cou * scale_cov < 1e-10:
                break

            scale_factor_u = sqrt(scale_cou / scale_cov / aspect)
            if axis != 'XY':
                scale_factor_u **= 2

            tolerance = 1e-5  # Trade accuracy for performance.
            if shear:
                t = Matrix.Identity(2)
                t[0][0] = scale_factor_u
                t[1][0] = clamp((scale_cross / isl.area_3d) * aspect, -0.5 * aspect, 0.5 * aspect)
                t[0][1] = 0
                t[1][1] = 1 / scale_factor_u

                if axis == 'X':
                    t[1][1] = 1
                    temp = t[0][1]
                    t[0][1] = t[1][0]
                    t[1][0] = temp

                elif axis == 'Y':
                    t[0][0] = 1

                err = abs(t[0][0] - 1.0) + abs(t[1][0]) + abs(t[0][1]) + abs(t[1][1] - 1.0)
                if err < tolerance:
                    break

                # Transform
                for uv_coord in isl.flat_unique_uv_coords:
                    uv_coord.xy = t @ uv_coord
                new_center = t @ new_center
            else:
                if math.isclose(scale_factor_u, 1.0, abs_tol=tolerance):
                    break
                scale = Vector((scale_factor_u, 1.0/scale_factor_u))
                if axis == 'X':
                    scale.y = 1
                elif axis == 'Y':
                    scale.x = 1

                for uv_coord in isl.flat_unique_uv_coords:
                    uv_coord *= scale
                new_center *= scale
            isl.umesh.update_tag = True
        return new_center


class UNIV_OT_Normalize_VIEW3D(Operator, utils.OverlapHelper):
    bl_idname = "mesh.univ_normalize"
    bl_label = 'Normalize'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Average the size of separate UV islands, based on their area in 3D space\n\n" \
                     f"Default - Average Islands Scale\n" \
                     f"Shift - Lock Overlaps"

    shear: BoolProperty(name='Shear', default=False, description='Reduce shear within islands')
    xy_scale: BoolProperty(name='Scale Independently', default=True, description='Scale U and V independently')
    use_aspect: BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.lock_overlap = event.shift
        return self.execute(context)

    def draw(self, context):
        layout = self.layout
        self.draw_overlap()
        layout.prop(self, 'shear')
        layout.prop(self, 'xy_scale')
        if hasattr(self, 'invert'):
            layout.prop(self, 'invert')
        layout.prop(self, 'use_aspect')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.umeshes = types.UMeshes(report=self.report)
        is_uv_area = context.area.ui_type == 'UV'
        if not is_uv_area:
            self.umeshes.set_sync(True)

        for umesh in self.umeshes:
            umesh.update_tag = False
            umesh.value = umesh.check_uniform_scale(report=self.report)

        all_islands: list[AdvIsland | UnionIslands] = []

        islands_calc_type: Callable[[types.UMesh], AdvIslands]
        if self.umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            self.umeshes = selected_umeshes if selected_umeshes else unselected_umeshes
            # TODO: AdvIslands with FLIPPED_3D
            islands_calc_type = AdvIslands.calc_extended_with_mark_seam if selected_umeshes else AdvIslands.calc_visible_with_mark_seam
        else:
            islands_calc_type = AdvIslands.calc_with_hidden
            for umesh in self.umeshes:
                umesh.ensure(face=True)

        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0
            adv_islands = islands_calc_type(umesh)
            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'
            all_islands.extend(adv_islands)
            adv_islands.calc_tris()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale

        if not all_islands:
            self.report({'WARNING'}, 'Islands not found')
            return {'CANCELLED'}

        if self.lock_overlap:
            all_islands = self.calc_overlapped_island_groups(all_islands)

        if self.xy_scale or self.shear:
            for isl in all_islands:
                isl.value = isl.bbox.center  # isl.value == pivot
                self.individual_scale(isl)

        tot_area_uv, tot_area_3d = self.avg_by_frequencies(all_islands)
        self.normalize(all_islands, tot_area_uv, tot_area_3d)

        self.umeshes.update(info='All islands were normalized')

        if not self.umeshes.is_edit_mode:
            self.umeshes.free()
            utils.update_area_by_type('VIEW_3D')

        return {'FINISHED'}

    def individual_scale(self, isl: AdvIsland):
        from bl_math import clamp
        aspect = isl.umesh.aspect
        shear = self.shear
        xy_scale = self.xy_scale

        vectors_ac_bc = [(va - vc, vb - vc) for va, vb, vc in isl.flat_3d_coords]
        uv_coords_and_3d_vectors_and_3d_areas = tuple(zip(isl.flat_coords, vectors_ac_bc, isl.weights))
        for j in range(10):
            scale_cou = 0.0
            scale_cov = 0.0
            scale_cross = 0.0

            for (uv_a, uv_b, uv_c), (vec_ac, vec_bc), weight in uv_coords_and_3d_vectors_and_3d_areas:
                if isclose(area_tri(uv_a, uv_b, uv_c), 0, abs_tol=1e-9):
                    continue
                m = Matrix((uv_a - uv_c, uv_b - uv_c))
                try:
                    m.invert()
                except ValueError:
                    continue

                cou = m[0][0] * vec_ac + m[0][1] * vec_bc
                cov = m[1][0] * vec_ac + m[1][1] * vec_bc

                scale_cou += cou.length * weight
                scale_cov += cov.length * weight

                if shear:
                    cov.normalize()
                    cou.normalize()
                    scale_cross += cou.dot(cov) * weight

            if scale_cou * scale_cov < 1e-10:
                break

            scale_factor_u = sqrt(scale_cou / scale_cov / aspect) if xy_scale else 1.0
            tolerance = 1e-5  # Trade accuracy for performance.
            if shear:
                t = Matrix.Identity(2)
                t[0][0] = scale_factor_u
                t[1][0] = clamp((scale_cross / isl.area_3d) * aspect, -0.5 * aspect, 0.5 * aspect)
                t[0][1] = 0
                t[1][1] = 1 / scale_factor_u

                err = abs(t[0][0] - 1.0) + abs(t[1][0]) + abs(t[0][1]) + abs(t[1][1] - 1.0)
                if err < tolerance:
                    break

                # Transform
                for uv_coord in isl.flat_unique_uv_coords:
                    uv_coord.xy = t @ uv_coord  # TODO: Calc new pivot from old pivot ond save in bbox
            else:
                if math.isclose(scale_factor_u, 1.0, abs_tol=tolerance):
                    break
                scale = Vector((scale_factor_u, 1.0/scale_factor_u))
                for uv_coord in isl.flat_unique_uv_coords:
                    uv_coord *= scale

            isl.umesh.update_tag = True

    def normalize(self, islands: list[AdvIsland], tot_area_uv, tot_area_3d):
        if not self.xy_scale and len(islands) <= 1:
            self.umeshes.cancel_with_report({'WARNING'}, info=f"Islands should be more than 1, given {len(islands)} islands")
            return
        if tot_area_3d == 0.0 or tot_area_uv == 0.0:
            # Prevent divide by zero.
            self.umeshes.cancel_with_report({'WARNING'}, info=f"Cannot normalize islands, total {'UV-area' if tot_area_3d else '3D-area'} of faces is zero")
            return

        tot_fac = tot_area_3d / tot_area_uv

        zero_area_islands = []
        for isl in islands:
            if isclose(isl.area_3d, 0.0, abs_tol=1e-6) or isclose(isl.area_uv, 0.0, abs_tol=1e-6):
                zero_area_islands.append(isl)
                continue

            fac = isl.area_3d / isl.area_uv
            scale = math.sqrt(fac / tot_fac)

            if self.xy_scale or self.shear:
                old_pivot = isl.value
                new_pivot = isl.calc_bbox().center
                new_pivot_with_scale = new_pivot * scale

                diff1 = old_pivot - new_pivot
                diff = (new_pivot - new_pivot_with_scale) + diff1

                if utils.vec_isclose(old_pivot, new_pivot) and math.isclose(scale, 1.0, abs_tol=0.00001):
                    continue

                for crn_co in isl.flat_unique_uv_coords:
                    crn_co *= scale
                    crn_co += diff

                isl.umesh.update_tag = True
            else:
                if math.isclose(scale, 1.0, abs_tol=0.00001):
                    continue
                if isl.scale(Vector((scale, scale)), pivot=isl.calc_bbox().center):
                    isl.umesh.update_tag = True

        if zero_area_islands:
            for isl in islands:
                if isl not in zero_area_islands:
                    isl.select = False
                    isl.umesh.update_tag = True
            for isl in zero_area_islands:
                isl.select = True
                isl.umesh.update_tag = True

            self.report({'WARNING'}, f"Found {len(zero_area_islands)} islands with zero area")

    def avg_by_frequencies(self, all_islands: list[AdvIsland]):
        areas_uv = np.empty(len(all_islands), dtype=float)
        areas_3d = np.empty(len(all_islands), dtype=float)

        for idx, isl in enumerate(all_islands):
            areas_uv[idx] = isl.calc_area_uv()
            areas_3d[idx] = isl.area_3d

        areas = areas_uv if self.bl_idname.startswith('UV') else areas_3d
        median: float = np.median(areas)  # noqa
        min_area = np.amin(areas)
        max_area = np.amax(areas)

        center = (min_area + max_area) / 2
        if center > median:
            diff = bl_math.lerp(median, max_area, 0.15) - median
        else:
            diff = median - bl_math.lerp(median, min_area, 0.15)

        min_clamp = median - diff
        max_clamp = median + diff

        indexes = (areas >= min_clamp) & (areas <= max_clamp)
        total_uv_area = np.sum(areas_uv, where=indexes)
        total_3d_area = np.sum(areas_3d, where=indexes)

        # TODO: Averaging by area_3d to area_uv ratio (by frequency of occurrence of the same values)
        if total_uv_area and total_3d_area:
            return total_uv_area, total_3d_area
        else:
            idx_for_find = math.nextafter(median, max_area)
            idx = UNIV_OT_Normalize_VIEW3D.np_find_nearest(areas, idx_for_find)
            total_uv_area = areas_uv[idx]
            total_3d_area = areas_3d[idx]
            if total_uv_area and total_3d_area:
                return total_uv_area, total_3d_area
            else:
                return np.sum(areas_uv), np.sum(areas_3d)

    @staticmethod
    def np_find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

class UNIV_OT_Normalize(UNIV_OT_Normalize_VIEW3D):
    bl_idname = "uv.univ_normalize"
    bl_description = UNIV_OT_Normalize_VIEW3D.bl_description + "\n\nHas [Shift + A] keymap"


class UNIV_OT_AdjustScale_VIEW3D(UNIV_OT_Normalize_VIEW3D):
    bl_idname = "mesh.univ_adjust_td"
    bl_label = 'Adjust'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = "Average the size of separate UV islands from unselected islands or objects, based on their area in 3D space\n\n" \
                     "Default - Average Islands Scale\n" \
                     "Shift - Lock Overlaps\n" \
                     "Ctrl or Alt - Invert"

    invert: BoolProperty(name='Invert', default=False)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if self.bl_idname.startswith('UV'):
            self.max_distance = utils.get_max_distance_from_px(prefs().max_pick_distance, context.region.view2d)
            self.mouse_pos = utils.get_mouse_pos(context, event)
        if event.value == 'PRESS':
            return self.execute(context)
        self.lock_overlap = event.shift
        self.invert = event.ctrl or event.alt
        return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mouse_pos = Vector((0, 0))
        self.max_distance: float | None = None

    def execute(self, context):
        if context.mode == 'EDIT_MESH':
            return self.adjust_edit()
        return self.adjust_object()

    def pick_adjust_edit(self):
        all_islands = []
        hit = types.IslandHit(self.mouse_pos, self.max_distance)
        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0
            adv_islands = AdvIslands.calc_visible_with_mark_seam(umesh)
            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'

            adv_islands.calc_tris()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_uv()
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale
            all_islands.extend(adv_islands)

        if self.lock_overlap:
            threshold = self.threshold if self.lock_overlap_mode == 'EXACT' else None
            all_islands = UnionIslands.calc_overlapped_island_groups(all_islands, threshold)

        for isl in all_islands:
            hit.find_nearest_island(isl)

        if not hit or (self.max_distance < hit.min_dist):
            self.report({'INFO'}, 'Island not found within a given radius')
            return {'CANCELLED'}

        all_islands.remove(hit.island)

        tot_area_uv = tot_area_3d = 0
        if self.invert:
            tot_area_uv += hit.island.area_uv
            tot_area_3d += hit.island.area_3d
        else:
            for isl in all_islands:
                tot_area_uv += isl.area_uv
                tot_area_3d += isl.area_3d
            all_islands = [hit.island]

        if self.xy_scale or self.shear:
            for isl in all_islands:
                isl.value = isl.bbox.center  # isl.value == pivot
                self.individual_scale(isl)

        self.show_adjust_result_info_edit(all_islands, tot_area_3d, tot_area_uv, sel='picked', unsel='unpicked')
        return {'FINISHED'}

    def adjust_edit(self):
        all_islands: list[AdvIsland | UnionIslands] = []
        self.umeshes = types.UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV') or not self.umeshes.is_edit_mode:
            self.umeshes.set_sync()

        for umesh in self.umeshes:
            umesh.update_tag = False
            umesh.value = umesh.check_uniform_scale(report=self.report)

        if self.invert:
            full_selected, not_full_selected = self.umeshes.filtered_by_full_selected_and_visible_uv_faces()
            if self.max_distance and not full_selected:
                if not_full_selected and all(not u.has_selected_uv_faces() for u in not_full_selected):
                    self.umeshes = not_full_selected
                    return self.pick_adjust_edit()

            unselected_umeshes = full_selected
            selected_umeshes = not_full_selected
            self.umeshes = selected_umeshes
            if not self.umeshes:
                self.report({'WARNING'}, 'Islands not found')
                return {'CANCELLED'}
        else:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if self.max_distance and not selected_umeshes and unselected_umeshes:
                self.umeshes = unselected_umeshes
                return self.pick_adjust_edit()

            self.umeshes = selected_umeshes

            if not self.umeshes:
                self.report({'WARNING'}, 'Islands not found')
                return {'CANCELLED'}

        tot_area_uv = tot_area_3d = 0
        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0
            adv_islands = AdvIslands.calc_visible_with_mark_seam(umesh)
            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'

            adv_islands.calc_tris()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_uv()
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale

            for isl in adv_islands:
                any_selected = AdvIslands.island_filter_is_any_face_selected(isl, umesh)
                if self.invert:
                    any_selected = not any_selected

                if any_selected:
                    all_islands.append(isl)
                else:
                    tot_area_uv += isl.area_uv
                    tot_area_3d += isl.area_3d

        for umesh in unselected_umeshes:
            if not (faces := utils.calc_visible_uv_faces(umesh)):
                continue
            adv_islands = AdvIsland(faces, umesh)
            tot_area_uv += adv_islands.calc_area_uv()
            tot_area_3d += adv_islands.calc_area_3d(scale=umesh.value)

        if self.lock_overlap:
            threshold = self.threshold if self.lock_overlap_mode == 'EXACT' else None
            all_islands = UnionIslands.calc_overlapped_island_groups(all_islands, threshold)

        if self.xy_scale or self.shear:
            for isl in all_islands:
                isl.value = isl.bbox.center  # isl.value == pivot
                self.individual_scale(isl)

        self.show_adjust_result_info_edit(all_islands, tot_area_3d, tot_area_uv)
        return {'FINISHED'}

    def show_adjust_result_info_edit(self, all_islands, tot_area_3d, tot_area_uv, sel='selected', unsel='unselected'):
        info_ = 'All target islands were normalized'
        if isinstance(tot_area_uv, int):
            if self.invert:
                sel, unsel = unsel, sel

            ret = self.umeshes.update(info=info_)
            if ret == {'FINISHED'}:
                for isl in all_islands:
                    isl.set_position(isl.value, isl.calc_bbox().center)
                self.report({'INFO'}, f'{unsel.capitalize()} islands not found, but {sel} was adjusted')
            else:
                self.report({'WARNING'}, f'{unsel.capitalize()} islands not found')
            return ret

        self.normalize(all_islands, tot_area_uv, tot_area_3d)
        self.umeshes.update(info=info_)

    def adjust_object(self):
        all_islands: list[AdvIsland | UnionIslands] = []
        self.umeshes = types.UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV') or not self.umeshes.is_edit_mode:
            self.umeshes.set_sync()

        for umesh in self.umeshes:
            umesh.update_tag = False
            umesh.value = umesh.check_uniform_scale(report=self.report)

        for umesh in (unselected_umeshes := types.UMeshes.unselected_with_uv()):
            umesh.value = umesh.check_uniform_scale(report=self.report)
        unselected_umeshes.set_sync()

        if self.invert:
            self.umeshes, unselected_umeshes = unselected_umeshes, self.umeshes

        tot_area_uv = tot_area_3d = 0
        for umesh in self.umeshes:
            umesh.aspect = utils.get_aspect_ratio(umesh) if self.use_aspect else 1.0  # TODO: Report heterogeneous aspects
            umesh.ensure()
            adv_islands = AdvIslands.calc_with_hidden(umesh)

            assert adv_islands, f'Object "{umesh.obj.name}" not found islands'

            adv_islands.calc_tris()
            adv_islands.calc_flat_uv_coords(save_triplet=True)
            adv_islands.calc_flat_unique_uv_coords()
            adv_islands.calc_flat_3d_coords(save_triplet=True, scale=umesh.value)
            adv_islands.calc_area_uv()
            adv_islands.calc_area_3d(umesh.value, areas_to_weight=True)  # umesh.value == obj scale
            all_islands.extend(adv_islands)

        for umesh in unselected_umeshes:
            adv_islands = AdvIsland(umesh.bm.faces, umesh)  # noqa
            tot_area_uv += adv_islands.calc_area_uv()
            tot_area_3d += adv_islands.calc_area_3d(scale=umesh.value)
            umesh.free()

        if self.lock_overlap:
            threshold = self.threshold if self.lock_overlap_mode == 'EXACT' else None
            all_islands = UnionIslands.calc_overlapped_island_groups(all_islands, threshold)

        if self.xy_scale or self.shear:
            for isl in all_islands:
                isl.value = isl.bbox.center  # isl.value == pivot
                self.individual_scale(isl)

        sel = 'selected'
        unsel = 'unselected'
        if self.invert:
            sel, unsel = unsel, sel

        self.umeshes.report_obj = None
        if isinstance(tot_area_uv, int):
            if (ret := self.umeshes.update()) == {'FINISHED'}:
                for isl in all_islands:
                    isl.set_position(isl.value, isl.calc_bbox().center)
                self.report({'INFO'}, f'{unsel.capitalize()} objects not found, but {sel} was adjusted')
            else:
                self.report({'WARNING'}, f"{unsel.capitalize()} objects not found")

            self.umeshes.free()
            utils.update_area_by_type('VIEW_3D')
            return ret

        self.normalize(all_islands, tot_area_uv, tot_area_3d)
        self.umeshes.report_obj = self.report

        if self.umeshes.has_update_mesh:
            if not unselected_umeshes:
                self.umeshes.update(info=f'{unsel.capitalize()} objects not found, but {sel} was adjusted')
            else:
                self.umeshes.update(info='All target islands were adjusted')
        else:
            self.report({'WARNING'}, f'{unsel.capitalize()} objects not found.')

        self.umeshes.free()
        utils.update_area_by_type('VIEW_3D')

        return {'FINISHED'}

class UNIV_OT_AdjustScale(UNIV_OT_AdjustScale_VIEW3D):
    bl_idname = "uv.univ_adjust_td"
    bl_description = UNIV_OT_AdjustScale_VIEW3D.bl_description + "\n\nHas [Alt + A] keymap, but it conflicts with the 'Deselect All' operator"


class UNIV_OT_Pack(Operator):
    bl_idname = 'uv.univ_pack'
    bl_label = 'Pack'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = f"Pack selected islands\n\n" \
                     f"Has [P] keymap, but it conflicts with the 'Pin' operator"

    def invoke(self, context, event):
        return self.execute(context)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
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


class UNIV_OT_TexelDensitySet_VIEW3D(Operator):
    bl_idname = "mesh.univ_texel_density_set"
    bl_label = 'Set TD'
    bl_description = "Set Texel Density"
    bl_options = {'REGISTER', 'UNDO'}

    grouping_type: EnumProperty(name='Grouping Type', default='NONE',
                                items=(('NONE', 'None', ''), ('OVERLAP', 'Overlap', ''), ('UNION', 'Union', '')))
    lock_overlap_mode: bpy.props.EnumProperty(name='Lock Overlaps Mode', default='ANY',
                                              items=(('ANY', 'Any', ''), ('EXACT', 'Exact', '')))
    threshold: bpy.props.FloatProperty(name='Distance', default=0.001, min=0.0, soft_min=0.00005, soft_max=0.00999)
    custom_texel: FloatProperty(name='Custom Texel', default=-1, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        if event.shift:
            self.grouping_type = 'UNION' if event.alt else 'OVERLAP'
        else:
            self.grouping_type = 'NONE'
        return self.execute(context)

    def draw(self, context):
        layout = self.layout  # noqa
        if self.grouping_type == 'OVERLAP':
            if self.lock_overlap_mode == 'EXACT':
                layout.prop(self, 'threshold', slider=True)
            layout.row().prop(self, 'lock_overlap_mode', expand=True)
        layout.row(align=True).prop(self, 'grouping_type', expand=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texel: float = 1.0
        self.texture_size: float = 2048.0
        self.has_selected = True
        self.islands_calc_type: Callable = Callable
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.texel = univ_settings().texel_density
        if self.custom_texel != -1.0:
            self.texel = bl_math.clamp(self.custom_texel, 1, 10_000)

        self.texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2
        self.umeshes = types.UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV') or not self.umeshes.is_edit_mode:
            self.umeshes.set_sync()

        cancel = False
        if not self.umeshes.is_edit_mode:
            if not self.umeshes:
                cancel = True
            else:
                self.has_selected = False
                self.islands_calc_type = AdvIslands.calc_with_hidden_with_mark_seam
                self.umeshes.ensure(True)
        else:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected_umeshes:
                self.has_selected = True
                self.umeshes = selected_umeshes
                self.islands_calc_type = AdvIslands.calc_extended_with_mark_seam
            elif unselected_umeshes:
                self.has_selected = False
                self.umeshes = unselected_umeshes
                self.islands_calc_type = AdvIslands.calc_visible_with_mark_seam
            else:
                cancel = True

        if cancel:
            self.report({'WARNING'}, 'Islands not found')
            return {'CANCELLED'}

        all_islands = []
        selected_islands_of_mesh = []
        zero_area_islands = []
        self.umeshes.update_tag = False

        for umesh in self.umeshes:
            if adv_islands := self.islands_calc_type(umesh):  # noqa
                umesh.value = umesh.check_uniform_scale(report=self.report)

                if self.grouping_type != 'NONE':
                    adv_islands.calc_tris()
                    adv_islands.calc_flat_uv_coords(save_triplet=True)
                    all_islands.extend(adv_islands)

                adv_islands.calc_area_uv()
                adv_islands.calc_area_3d(scale=umesh.value)

                if self.grouping_type == 'NONE':
                    for isl in adv_islands:
                        if (status := isl.set_texel(self.texel, self.texture_size)) is None:
                            zero_area_islands.append(isl)
                            continue
                        isl.umesh.update_tag |= status

                if self.has_selected:
                    selected_islands_of_mesh.append(adv_islands)

        if self.grouping_type != 'NONE':
            if self.grouping_type == 'OVERLAP':
                threshold = None if self.lock_overlap_mode == 'ANY' else self.threshold
                groups_of_islands = types.UnionIslands.calc_overlapped_island_groups(all_islands, threshold)
                for isl in groups_of_islands:
                    if (status := isl.set_texel(self.texel, self.texture_size)) is None:
                        zero_area_islands.append(isl)
                        continue
                    isl.umesh.update_tag |= status
            else:
                union_islands = types.UnionIslands(all_islands)
                status = union_islands.set_texel(self.texel, self.texture_size)
                union_islands.umesh.update_tag = status in (True, None)

                for u_isl in union_islands:
                    area_3d = sqrt(u_isl.area_3d)
                    area_uv = sqrt(u_isl.area_uv) * self.texture_size
                    if isclose(area_3d, 0.0, abs_tol=1e-6) or isclose(area_uv, 0.0, abs_tol=1e-6):
                        zero_area_islands.append(union_islands)

        if zero_area_islands:
            self.report({'WARNING'}, f"Found {len(zero_area_islands)} islands with zero area")
            if self.umeshes.is_edit_mode:
                for islands in selected_islands_of_mesh:
                    for isl in islands:
                        isl.select = False
                for isl in zero_area_islands:
                    isl.select = True
            self.umeshes.update_tag = True
            self.umeshes.silent_update()
            if not self.umeshes.is_edit_mode:
                self.umeshes.free()
                utils.update_area_by_type('VIEW_3D')
            return {'FINISHED'}

        if not self.umeshes.is_edit_mode:
            self.umeshes.update(info='All islands adjusted')
            self.umeshes.free()
            if self.umeshes.update_tag:
                utils.update_area_by_type('VIEW_3D')
            return {'FINISHED'}
        self.umeshes.update(info='All islands adjusted')
        return {'FINISHED'}

class UNIV_OT_TexelDensitySet(UNIV_OT_TexelDensitySet_VIEW3D):
    bl_idname = "uv.univ_texel_density_set"

class UNIV_OT_TexelDensityGet_VIEW3D(Operator):
    bl_idname = "mesh.univ_texel_density_get"
    bl_label = 'Get TD'
    bl_description = "Get Texel Density"

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.texel: float = 1.0
        self.texture_size: float = 2048.0
        self.has_selected = True
        self.umeshes: types.UMeshes | None = None

    def execute(self, context):
        self.texel = univ_settings().texel_density
        self.texture_size = (int(univ_settings().size_x) + int(univ_settings().size_y)) / 2
        self.umeshes = types.UMeshes(report=self.report)

        if not self.bl_idname.startswith('UV') or not self.umeshes.is_edit_mode:
            self.umeshes.set_sync()

        cancel = False
        if self.umeshes.is_edit_mode:
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_faces()
            if selected_umeshes:
                self.has_selected = True
                self.umeshes = selected_umeshes
            elif unselected_umeshes:
                self.has_selected = False
                self.umeshes = unselected_umeshes
            else:
                cancel = True
        else:
            if not self.umeshes:
                cancel = True
            else:
                self.has_selected = False

        if cancel:
            self.report({'WARNING'}, 'Faces not found')
            return {'CANCELLED'}

        total_3d_area = 0.0
        total_uv_area = 0.0

        for umesh in self.umeshes:
            if self.umeshes.is_edit_mode:
                faces = utils.calc_uv_faces(umesh, selected=self.has_selected)
            else:
                faces = umesh.bm.faces
            scale = umesh.check_uniform_scale(self.report)
            total_3d_area += utils.calc_total_area_3d(faces, scale)
            total_uv_area += utils.calc_total_area_uv(faces, umesh.uv)

        area_3d = sqrt(total_3d_area)
        area_uv = sqrt(total_uv_area) * self.texture_size
        if isclose(area_3d, 0.0, abs_tol=1e-6) or isclose(area_uv, 0.0, abs_tol=1e-6):
            self.report({'WARNING'}, f"All faces has zero area")
            return {'CANCELLED'}
        texel = area_uv / area_3d
        univ_settings().texel_density = bl_math.clamp(texel, 1.0, 10_000.0)
        return {'FINISHED'}

class UNIV_OT_TexelDensityGet(UNIV_OT_TexelDensityGet_VIEW3D):
    bl_idname = "uv.univ_texel_density_get"
