import bpy
import math
import random
import bl_math

import numpy as np

from bpy.types import Operator
from bpy.props import *

from math import pi
from mathutils import Vector

from ..types import BBox, Islands, AdvIslands, AdvIsland, FaceIsland, UnionIslands
from .. import utils
from .. import info


class UNIV_OT_Crop(Operator):
    bl_idname = 'uv.univ_crop'
    bl_label = 'Crop'
    bl_description = info.operator.crop_info
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='DEFAULT', items=(
        ('DEFAULT', 'Default', ''),
        ('TO_CURSOR', 'To cursor', ''),
        ('TO_CURSOR_INDIVIDUAL', 'To cursor individual', ''),
        ('INDIVIDUAL', 'Individual', ''),
        ('INPLACE', 'Inplace', ''),
        ('INDIVIDUAL_INPLACE', 'Individual Inplace', ''),
    ))

    axis: EnumProperty(name='Axis', default='XY', items=(('XY', 'XY', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    padding: FloatProperty(name='Padding', description='Padding=1/TextureSize (1/256=0.0039)', default=0, soft_min=0, soft_max=1/256*4, max=0.49)

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        match event.ctrl, event.shift, event.alt:
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
            case _:
                self.report({'INFO'}, f"Event: {info.event_to_string(event)} not implement. \n\n"
                                      f"See all variations:\n\n{self.get_event_info()}")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        sync = context.scene.tool_settings.use_uv_select_sync
        return self.crop(self.mode, self.axis, self.padding, proportional=True, sync=sync, report=self.report)

    @staticmethod
    def crop(mode, axis, padding, proportional, sync, report=None):
        umeshes = utils.UMeshes(report=report)
        crop_args = [axis, padding, umeshes, proportional, sync]
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
    def crop_default(axis, padding, umeshes, proportional, sync, offset=Vector((0, 0)), inplace=False, extended=True):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
            umesh.update_tag = bool(islands)

        if not islands_of_mesh:
            return

        UNIV_OT_Crop.crop_ex(axis, general_bbox, inplace, islands_of_mesh, offset, padding, proportional)

    @staticmethod
    def crop_individual(axis, padding, umeshes, proportional, sync, offset=Vector((0, 0)), inplace=False, extended=True):
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    UNIV_OT_Crop.crop_ex(axis, island.calc_bbox(), inplace, (island, ), offset, padding, proportional)
            umesh.update_tag = bool(islands)

    @staticmethod
    def crop_inplace(axis, padding, umeshes, proportional, sync, inplace=True, extended=True):
        islands_of_tile: dict[int | list[tuple[FaceIsland | BBox]]] = {}
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
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
        sync = context.scene.tool_settings.use_uv_select_sync
        return self.crop(self.mode, self.axis, self.padding, proportional=False, sync=sync, report=self.report)

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

    mode: EnumProperty(name="Mode", default='MOVE', items=(
        ('ALIGN', 'Align', ''),
        ('MOVE', 'Move', ''),
        ('ALIGN_CURSOR', 'Move cursor to selected', ''),
        ('ALIGN_TO_CURSOR', 'Align to cursor', ''),
        ('ALIGN_TO_CURSOR_UNION', 'Align to cursor union', ''),
        ('CURSOR_TO_TILE', 'Align cursor to tile', ''),
        ('MOVE_CURSOR', 'Move cursor', ''),
        # ('MOVE_COLLISION', 'Collision move', '')
    ))

    direction: EnumProperty(name="Direction", default='UPPER', items=align_align_direction_items)

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

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
            case False, True, True:
                self.mode = 'MOVE_CURSOR'
            case False, True, False:
                self.mode = 'MOVE'
            case _:
                self.report({'INFO'}, f"Event: {info.event_to_string(event)} not implement. \n\n"
                                      f"See all variations:\n\n{info.operator.align_event_info_ex}")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        return self.align(self.mode, self.direction, sync=context.scene.tool_settings.use_uv_select_sync, report=self.report)

    @staticmethod
    def align(mode, direction, sync, report=None):
        umeshes = utils.UMeshes(report=report)

        match mode:
            case 'ALIGN':
                UNIV_OT_Align.align_ex(direction, sync,  umeshes, selected=True)
                if not umeshes.final():
                    UNIV_OT_Align.align_ex(direction, sync,  umeshes,  selected=False)

            case 'ALIGN_TO_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Align.move_to_cursor_ex(cursor_loc, direction, umeshes, sync, selected=True)
                if not umeshes.final():
                    UNIV_OT_Align.move_to_cursor_ex(cursor_loc, direction, umeshes, sync, selected=False)

            case 'ALIGN_TO_CURSOR_UNION':
                if not (cursor_loc := utils.get_cursor_location()):
                    umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Align.move_to_cursor_union_ex(cursor_loc, direction, umeshes, sync, selected=True)
                if not umeshes.final():
                    UNIV_OT_Align.move_to_cursor_union_ex(cursor_loc, direction, umeshes, sync, selected=False)

            case 'ALIGN_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                general_bbox = UNIV_OT_Align.align_cursor_ex(umeshes, sync, selected=True)
                if not general_bbox.is_valid:
                    general_bbox = UNIV_OT_Align.align_cursor_ex(umeshes, sync, selected=False)
                if not general_bbox.is_valid:
                    umeshes.report()
                    return {'CANCELLED'}
                UNIV_OT_Align.align_cursor(direction, general_bbox, cursor_loc)
                return {'FINISHED'}

            case 'CURSOR_TO_TILE':
                if not (cursor_loc := utils.get_cursor_location()):
                    umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Align.align_cursor_to_tile(direction, cursor_loc)
                return {'FINISHED'}

            case 'MOVE_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    umeshes.report({'INFO'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Align.move_cursor(direction, cursor_loc)
                return {'FINISHED'}

            case 'MOVE':
                UNIV_OT_Align.move_ex(direction, sync, umeshes, selected=True)
                if not umeshes.final():
                    UNIV_OT_Align.move_ex(direction, sync, umeshes, selected=False)

            case _:
                raise NotImplementedError(mode)

        return umeshes.update()

    @staticmethod
    def move_to_cursor_ex(cursor_loc, direction, umeshes, sync, selected=True):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        island_mode = utils.is_island_mode()
        general_bbox = BBox.init_from_minmax(cursor_loc, cursor_loc)
        for umesh in umeshes:
            if island_mode:
                if islands := Islands.calc(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        all_groups.append((island, bbox, umesh.uv_layer))
                umesh.update_tag = bool(islands)
            else:
                if corners := utils.calc_uv_corners(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    all_groups.append((corners, umesh.uv_layer))
                umesh.update_tag = bool(corners)
        if island_mode:
            UNIV_OT_Align.align_islands(all_groups, direction, general_bbox, invert=True)
        else:  # Vertices or Edges UV selection mode
            UNIV_OT_Align.align_corners(all_groups, direction, general_bbox)

    @staticmethod
    def move_to_cursor_union_ex(cursor_loc, direction, umeshes, sync, selected=True):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        target_bbox = BBox.init_from_minmax(cursor_loc, cursor_loc)
        general_bbox = BBox()
        for umesh in umeshes:
            if faces := utils.calc_uv_faces(umesh.bm, umesh.uv_layer, sync, selected=selected):
                island = FaceIsland(faces, umesh.bm, umesh.uv_layer)
                bbox = island.calc_bbox()
                general_bbox.union(bbox)
                all_groups.append([island, bbox, umesh.uv_layer])
            umesh.update_tag = bool(faces)
        for group in all_groups:
            group[1] = general_bbox
        UNIV_OT_Align.align_islands(all_groups, direction, target_bbox, invert=True)

    @staticmethod
    def align_cursor_ex(umeshes, sync, selected):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        general_bbox = BBox()
        for umesh in umeshes:
            if corners := utils.calc_uv_corners(umesh.bm, umesh.uv_layer, sync, selected=selected):
                all_groups.append((corners, umesh.uv_layer))
                bbox = BBox.calc_bbox_uv_corners(corners, umesh.uv_layer)
                general_bbox.union(bbox)
        return general_bbox

    @staticmethod
    def align_ex(direction, sync, umeshes, selected=True):
        all_groups = []  # islands, bboxes, uv_layer or corners, uv_layer
        general_bbox = BBox()
        island_mode = utils.is_island_mode()
        for umesh in umeshes:
            if island_mode:
                if islands := Islands.calc(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    for island in islands:
                        bbox = island.calc_bbox()
                        general_bbox.union(bbox)

                        all_groups.append((island, bbox, umesh.uv_layer))
                umesh.update_tag = bool(islands)
            else:
                if corners := utils.calc_uv_corners(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    bbox = BBox.calc_bbox_uv_corners(corners, umesh.uv_layer)
                    general_bbox.union(bbox)

                    all_groups.append((corners, umesh.uv_layer))
                umesh.update_tag = bool(corners)
        if island_mode:
            UNIV_OT_Align.align_islands(all_groups, direction, general_bbox)
        else:  # Vertices or Edges UV selection mode
            UNIV_OT_Align.align_corners(all_groups, direction, general_bbox)  # TODO Individual ALign for Vertical and Horizontal or all

    @staticmethod
    def move_ex(direction, sync, umeshes, selected=True):
        island_mode = utils.is_island_mode()
        for umesh in umeshes:
            if island_mode:
                if islands := Islands.calc(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    match direction:
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
                            move_value = Vector(UNIV_OT_Align.get_move_value(direction))
                            for island in islands:
                                island.move(move_value)
                umesh.update_tag = bool(islands)
            else:
                if corners := utils.calc_uv_corners(umesh.bm, umesh.uv_layer, sync, selected=selected):
                    match direction:
                        case 'CENTER':
                            for corner in corners:
                                corner[umesh.uv_layer].uv = 0.5, 0.5
                        case 'HORIZONTAL':
                            for corner in corners:
                                corner[umesh.uv_layer].uv.x = 0.5
                        case 'VERTICAL':
                            for corner in corners:
                                corner[umesh.uv_layer].uv.y = 0.5
                        case _:
                            move_value = Vector(UNIV_OT_Align.get_move_value(direction))
                            for corner in corners:
                                corner[umesh.uv_layer].uv += move_value
                umesh.update_tag = bool(corners)

    @staticmethod
    def align_islands(groups, direction, general_bbox, invert=False):
        for island, bounds, uv_layer in groups:
            center = bounds.center
            match direction:
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
                    raise NotImplementedError(direction)
            island.move(Vector(delta))

    @staticmethod
    def align_corners(groups, direction, general_bbox):

        match direction:
            case 'LEFT' | 'RIGHT' | 'VERTICAL':
                if direction == 'LEFT':
                    destination = general_bbox.min.x
                elif direction == 'RIGHT':
                    destination = general_bbox.max.x
                else:
                    destination = general_bbox.center.x

                for luvs, uv_layer in groups:
                    for luv in luvs:
                        luv[uv_layer].uv[0] = destination
            case 'UPPER' | 'BOTTOM' | 'HORIZONTAL':
                if direction == 'UPPER':
                    destination = general_bbox.max.y
                elif direction == 'BOTTOM':
                    destination = general_bbox.min.y
                else:
                    destination = general_bbox.center.y

                for luvs, uv_layer in groups:
                    for luv in luvs:
                        luv[uv_layer].uv[1] = destination
            case _:
                if direction == 'CENTER':
                    destination = general_bbox.center
                elif direction == 'LEFT_BOTTOM':
                    destination = general_bbox.left_bottom
                elif direction == 'RIGHT_UPPER':
                    destination = general_bbox.right_upper
                elif direction == 'LEFT_UPPER':
                    destination = general_bbox.left_upper
                elif direction == 'RIGHT_BOTTOM':
                    destination = general_bbox.right_bottom
                else:
                    raise NotImplementedError(direction)

                for luvs, uv_layer in groups:
                    for luv in luvs:
                        luv[uv_layer].uv = destination

    @staticmethod
    def align_cursor(direction: str, general_bbox, cursor_loc):

        if direction in ('UPPER', 'BOTTOM'):
            loc = getattr(general_bbox, direction.lower())
            loc.x = cursor_loc.x
            utils.set_cursor_location(loc)
        elif direction in ('RIGHT', 'LEFT'):
            loc = getattr(general_bbox, direction.lower())
            loc.y = cursor_loc.y
            utils.set_cursor_location(loc)
        elif loc := getattr(general_bbox, direction.lower(), False):
            utils.set_cursor_location(loc)
        elif direction == 'VERTICAL':
            utils.set_cursor_location(Vector((general_bbox.center.x, cursor_loc.y)))
        elif direction == 'HORIZONTAL':
            utils.set_cursor_location(Vector((cursor_loc.x, general_bbox.center.y)))
        else:
            raise NotImplementedError(direction)

    @staticmethod
    def align_cursor_to_tile(direction: str, cursor_loc):
        def pad_floor(value):
            f = np.floor(np.float16(value))
            return np.nextafter(f, f + np.float16(1.0))

        def pad_ceil(value):
            f = np.floor(np.float16(value))
            return np.nextafter(f + np.float16(1.0), f)

        x, y = cursor_loc
        match direction:
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
                raise NotImplementedError(direction)
        utils.set_cursor_location((x, y))

    @staticmethod
    def move_cursor(direction: str, cursor_loc):
        match direction:
            case 'CENTER' | 'HORIZONTAL' | 'VERTICAL':
                UNIV_OT_Align.align_cursor_to_tile(direction, cursor_loc)
            case _:
                delta = Vector(UNIV_OT_Align.get_move_value(direction))
                utils.set_cursor_location(cursor_loc+delta)

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
        return

class UNIV_OT_Flip(Operator):
    bl_idname = 'uv.univ_flip'
    bl_label = 'Flip'
    bl_description = 'FlipX and FlipY'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='DEFAULT', items=(
        ('DEFAULT', 'Default', ''),
        ('BY_CURSOR', 'By cursor', ''),
        ('INDIVIDUAL', 'Individual', ''),
        ('FLIPPED', 'Flipped', ''),
        ('FLIPPED_INDIVIDUAL', 'Flipped Individual', ''),
    ))

    axis: EnumProperty(name='Axis', default='X', items=(('X', 'X', ''), ('Y', 'Y', '')))

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        match event.ctrl, event.shift, event.alt:
            case False, False, False:
                self.mode = 'DEFAULT'
            case True, False, False:
                self.mode = 'BY_CURSOR'
            case False, True, False:
                self.mode = 'INDIVIDUAL'
            case False, False, True:
                self.mode = 'FLIPPED'
            case False, True, True:
                self.mode = 'FLIPPED_INDIVIDUAL'
            case _:
                self.report({'INFO'}, f"Event: {info.event_to_string(event)} not implement. \n\n"
                                      f"See all variations:\n\n")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        return self.flip(self.mode, self.axis, sync=context.scene.tool_settings.use_uv_select_sync, report=self.report)

    @staticmethod
    def flip(mode, axis, sync, report=None):
        umeshes = utils.UMeshes(report=report)
        flip_args = (axis, sync,  umeshes)

        match mode:
            case 'DEFAULT':
                UNIV_OT_Flip.flip_ex(*flip_args, extended=True)
                if not umeshes.final():
                    UNIV_OT_Flip.flip_ex(*flip_args, extended=False)

            case 'BY_CURSOR':
                if not (cursor_loc := utils.get_cursor_location()):
                    umeshes.report({'INFO'}, "Cursor not found")
                UNIV_OT_Flip.flip_by_cursor(*flip_args, cursor=cursor_loc, extended=True)
                if not umeshes.final():
                    UNIV_OT_Flip.flip_by_cursor(*flip_args, cursor=cursor_loc, extended=False)

            case 'INDIVIDUAL':
                UNIV_OT_Flip.flip_individual(*flip_args, extended=True)
                if not umeshes.final():
                    UNIV_OT_Flip.flip_individual(*flip_args, extended=False)

            case 'FLIPPED':
                UNIV_OT_Flip.flip_flipped(*flip_args, extended=True)
                if not umeshes.final():
                    UNIV_OT_Flip.flip_flipped(*flip_args, extended=False)

            case 'FLIPPED_INDIVIDUAL':
                UNIV_OT_Flip.flip_flipped_individual(*flip_args, extended=True)
                if not umeshes.final():
                    UNIV_OT_Flip.flip_flipped_individual(*flip_args, extended=False)
            case _:
                raise NotImplementedError(mode)

        return umeshes.update()

    @staticmethod
    def flip_ex(axis, sync,  umeshes,  extended):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
            umesh.update_tag = bool(islands)

        if not islands_of_mesh:
            return

        pivot = general_bbox.center
        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        for islands in islands_of_mesh:
            islands.scale(scale=scale, pivot=pivot)

    @staticmethod
    def flip_by_cursor(axis, sync,  umeshes,  cursor, extended):
        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                islands.scale(scale=scale, pivot=cursor)
            umesh.update_tag = bool(islands)

    @staticmethod
    def flip_individual(axis, sync,  umeshes,  extended):
        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    island.scale(scale=scale, pivot=island.calc_bbox().center)
            umesh.update_tag = bool(islands)

    @staticmethod
    def flip_flipped(axis, sync,  umeshes, extended):
        flipped_islands_of_mesh = []
        general_bbox = BBox()
        umeshes_for_update = []
        has_islands = False
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                flipped_islands = Islands([isl for isl in islands if isl.is_flipped()], umesh.bm, umesh.uv_layer)
                if flipped_islands:
                    general_bbox.union(flipped_islands.calc_bbox())
                    flipped_islands_of_mesh.append(flipped_islands)
                    umeshes_for_update.append(umesh)
            umesh.update_tag = bool(islands)
            has_islands |= bool(islands)

        if umeshes_for_update and has_islands:
            for umesh in umeshes:
                umesh.update_tag = umesh in umeshes_for_update

        if not has_islands:
            return

        if has_islands and len(flipped_islands_of_mesh) == 0:
            return umeshes.cancel_with_report(info='Flipped islands not found')
        umeshes.report(info=f'Found {sum(len(f_isl) for f_isl in flipped_islands_of_mesh)} Flipped islands')

        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        pivot = general_bbox.center
        for islands in flipped_islands_of_mesh:
            islands.scale(scale, pivot)

    @staticmethod
    def flip_flipped_individual(axis, sync,  umeshes,  extended):
        scale = UNIV_OT_Flip.get_flip_scale_from_axis(axis)
        umeshes_for_update = []
        has_islands = False
        has_flipped_islands = False
        islands_count = 0
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                flipped_islands = [isl for isl in islands if isl.is_flipped()]
                for island in flipped_islands:
                    island.scale(scale=scale, pivot=island.calc_bbox().center)
                if flipped_islands:
                    umeshes_for_update.append(umesh)
                has_flipped_islands |= bool(flipped_islands)
                islands_count += len(flipped_islands)
            umesh.update_tag = bool(islands)
            has_islands |= bool(islands)

        if umeshes_for_update and has_islands:
            for umesh in umeshes:
                umesh.update_tag = umesh in umeshes_for_update

        if has_islands and not has_flipped_islands:
            return umeshes.cancel_with_report(info='Flipped islands not found')
        umeshes.report(info=f'Found {islands_count} Flipped islands')

    @staticmethod
    def get_flip_scale_from_axis(axis):
        return Vector((-1, 1)) if axis == 'X' else Vector((1, -1))


class UNIV_OT_Rotate(Operator):
    bl_idname = 'uv.univ_rotate'
    bl_label = 'Rotate'
    bl_description = 'Rotate CW and Rotate CCW'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode',
                       default='DEFAULT',
                       items=(('DEFAULT', 'Default', ''),
                              ('INDIVIDUAL', 'Individual', ''),
                              ('BY_CURSOR', 'By Cursor', ''))
                       )
    rot_dir: EnumProperty(name='Direction of rotation', default='CW', items=(('CW', 'CW', ''), ('CCW', 'CCW', '')))
    angle: FloatProperty(name='Angle', default=pi*0.5, min=0, max=pi, soft_min=math.radians(5.0), subtype='ANGLE')

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.rot_dir = 'CCW' if event.alt else 'CW'
        if event.shift:
            self.mode = 'INDIVIDUAL'
        elif event.ctrl:
            self.mode = 'BY_CURSOR'
        else:
            self.mode = 'DEFAULT'
        return self.execute(context)

    def execute(self, context):
        return self.rotate(sync=context.scene.tool_settings.use_uv_select_sync, report=self.report)

    def rotate(self, sync, report=None):
        umeshes = utils.UMeshes(report=report)
        angle = (-self.angle) if self.rot_dir == 'CCW' else self.angle
        flip_args = (angle, sync, umeshes)

        if self.mode == 'DEFAULT':
            UNIV_OT_Rotate.rotate_ex(*flip_args, extended=True)
            if not umeshes.final():
                UNIV_OT_Rotate.rotate_ex(*flip_args, extended=False)

        elif self.mode == 'BY_CURSOR':
            if not (cursor_loc := utils.get_cursor_location()):
                if report:
                    report({'INFO'}, "Cursor not found")
                return {'CANCELLED'}
            UNIV_OT_Rotate.rotate_by_cursor(*flip_args, cursor=cursor_loc, extended=True)
            if not umeshes.final():
                UNIV_OT_Rotate.rotate_by_cursor(*flip_args, cursor=cursor_loc, extended=False)

        elif self.mode == 'INDIVIDUAL':
            UNIV_OT_Rotate.rotate_individual(*flip_args, extended=True)
            if not umeshes.final():
                UNIV_OT_Rotate.rotate_individual(*flip_args, extended=False)
        else:
            raise NotImplementedError()

        return umeshes.update()

    @staticmethod
    def rotate_ex(angle, sync,  umeshes,  extended):
        islands_of_mesh = []
        general_bbox = BBox()
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                general_bbox.union(islands.calc_bbox())
                islands_of_mesh.append(islands)
            umesh.update_tag = bool(islands)

        pivot = general_bbox.center
        for islands in islands_of_mesh:
            islands.rotate(angle, pivot=pivot)

    @staticmethod
    def rotate_by_cursor(angle, sync,  umeshes, cursor, extended):
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                islands.rotate(angle, pivot=cursor)
            umesh.update_tag = bool(islands)

    @staticmethod
    def rotate_individual(angle, sync,  umeshes,  extended):
        for umesh in umeshes:
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    island.rotate(angle, pivot=island.calc_bbox().center)
            umesh.update_tag = bool(islands)


class UNIV_OT_Sort(Operator):
    bl_idname = 'uv.univ_sort'
    bl_label = 'Sort'
    bl_description = 'Sort'
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(name='Axis', default='AUTO', items=(('AUTO', 'Auto', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    padding: FloatProperty(name='Padding', default=1/2048, min=0, soft_max=0.1,)
    reverse: BoolProperty(name='Reverse', default=True)
    to_cursor: BoolProperty(name='To Cursor', default=False)
    align: BoolProperty(name='Align', default=False)
    overlapped: BoolProperty(name='Overlapped', default=False)

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.to_cursor = event.ctrl
        self.overlapped = event.shift
        self.align = event.alt
        return self.execute(context)

    def execute(self, context):
        return self.sort()

    def sort(self):
        cursor_loc = None
        if self.to_cursor:
            if not (cursor_loc := utils.get_cursor_location()):
                self.report({'INFO'}, "Cursor not found")
                return {'CANCELLED'}

        sync = bpy.context.scene.tool_settings.use_uv_select_sync
        umeshes = utils.UMeshes(report=self.report)
        sort_args = (sync,  umeshes, cursor_loc)

        if not self.overlapped:
            self.sort_individual_preprocessing(*sort_args, extended=True)
            if not umeshes.final():
                self.sort_individual_preprocessing(*sort_args, extended=False)
        else:
            self.sort_overlapped_preprocessing(*sort_args, extended=True)
            if not umeshes.final():
                self.sort_overlapped_preprocessing(*sort_args, extended=False)

        return umeshes.update()

    def sort_overlapped_preprocessing(self, sync, umeshes, cursor=None, extended=True):
        _islands: list[AdvIsland] = []
        for umesh in umeshes:
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                adv_islands.calc_tris()
                adv_islands.calc_flat_coords()
                _islands.extend(adv_islands)
            umesh.update_tag = bool(adv_islands)

        if not _islands:
            return

        general_bbox = BBox()
        union_islands_groups = UnionIslands.calc_overlapped_island_groups(_islands)
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

        self.sort_islands(cursor, general_bbox, umeshes, union_islands_groups)

    def sort_individual_preprocessing(self, sync, umeshes, cursor=None, extended=True):
        _islands: list[AdvIsland] = []
        general_bbox = BBox()
        for umesh in umeshes:
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
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
                _islands.extend(adv_islands)

            umesh.update_tag = bool(adv_islands)

        if not _islands:
            return
        self.sort_islands(cursor, general_bbox, umeshes, _islands)

    def sort_islands(self, cursor, general_bbox, umeshes, islands: list[AdvIsland | UnionIslands]):
        islands.sort(key=lambda x: x.bbox.max_length, reverse=self.reverse)
        if self.axis == 'AUTO':
            horizontal_sort = general_bbox.width * 2 > general_bbox.height
        else:
            horizontal_sort = self.axis == 'X'

        margin = general_bbox.min if (cursor is None) else cursor
        update_tag = False
        if horizontal_sort:
            for island in islands:
                width = island.bbox.width
                if self.align and island.bbox.height < width:
                    width = island.bbox.height
                    update_tag |= island.rotate(pi * 0.5, island.bbox.center)
                    island.calc_bbox()
                update_tag |= island.set_position(margin, _from=island.bbox.min)
                margin.x += self.padding + width
        else:
            for island in islands:
                height = island.bbox.height
                if self.align and island.bbox.width < height:
                    height = island.bbox.width
                    update_tag |= island.rotate(pi * 0.5, island.bbox.center)
                    island.calc_bbox()  # TODO: Optimize this
                update_tag |= island.set_position(margin, _from=island.bbox.min)
                margin.y += self.padding + height
        if not update_tag:
            umeshes.cancel_with_report(info='Islands is sorted')


class UNIV_OT_Distribute(Operator):
    bl_idname = 'uv.univ_distribute'
    bl_label = 'Distribute'
    bl_description = 'Distribute'
    bl_options = {'REGISTER', 'UNDO'}

    axis: EnumProperty(name='Axis', default='AUTO', items=(('AUTO', 'Auto', ''), ('X', 'X', ''), ('Y', 'Y', '')))
    padding: FloatProperty(name='Padding', default=1/2048, min=0, soft_max=0.1,)
    to_cursor: BoolProperty(name='To Cursor', default=False)
    overlapped: BoolProperty(name='Overlapped', default=False)
    space: BoolProperty(name='Space', default=False)

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.to_cursor = event.ctrl
        self.overlapped = event.shift
        self.space = event.alt
        return self.execute(context)

    def execute(self, context):
        return self.distribute(sync=context.scene.tool_settings.use_uv_select_sync)

    def distribute(self, sync):
        umeshes = utils.UMeshes(report=self.report)
        cursor_loc = None
        if self.to_cursor:
            if not (cursor_loc := utils.get_cursor_location()):
                umeshes.report({'INFO'}, "Cursor not found")
                return {'CANCELLED'}

        distribute_args = (sync,  umeshes, cursor_loc)

        if self.space:
            self.distribute_space(*distribute_args, extended=True)
            if not umeshes.final():
                self.distribute_space(*distribute_args, extended=False)
        else:
            self.distribute_ex(*distribute_args, extended=True)
            if not umeshes.final():
                self.distribute_ex(*distribute_args, extended=False)
        return umeshes.update()

    def distribute_ex(self, sync,  umeshes, cursor, extended=True):
        func = self.distribute_preprocessing_overlap if self.overlapped else self.distribute_preprocessing
        _islands, general_bbox = func(sync, umeshes, extended)

        if len(_islands) <= 2:
            if len(_islands) != 0:
                umeshes.cancel_with_report(info=f"The number of islands must be greater than two, {len(_islands)} was found")
            return

        if self.axis == 'AUTO':
            horizontal_distribute = general_bbox.width * 2 > general_bbox.height
        else:
            horizontal_distribute = self.axis == 'X'

        update_tag = False
        cursor_offset = 0
        if horizontal_distribute:
            _islands.sort(key=lambda a: a.bbox.xmin)
            if cursor is None:
                margin = general_bbox.min.x
            else:
                margin = cursor.x
                cursor_offset += general_bbox.min.y - cursor.y

            for island in _islands:
                width = island.bbox.width
                update_tag |= island.set_position(Vector((margin, island.bbox.ymin - cursor_offset)), _from=island.bbox.min)
                margin += self.padding + width
        else:
            _islands.sort(key=lambda a: a.bbox.ymin)
            if cursor is None:
                margin = general_bbox.min.y
            else:
                margin = cursor.y
                cursor_offset += general_bbox.min.x - cursor.x

            for island in _islands:
                height = island.bbox.height
                update_tag |= island.set_position(Vector((island.bbox.xmin - cursor_offset, margin)), _from=island.bbox.min)
                margin += self.padding + height
        if not update_tag:
            umeshes.cancel_with_report(info='Islands is Distributed')

    def distribute_space(self, sync, umeshes, cursor, extended=True):
        func = self.distribute_preprocessing_overlap if self.overlapped else self.distribute_preprocessing
        _islands, general_bbox = func(sync, umeshes, extended)

        if len(_islands) <= 2:
            if len(_islands) != 0:
                umeshes.cancel_with_report(info=f"The number of islands must be greater than two, {len(_islands)} was found")
            return

        if self.axis == 'AUTO':
            horizontal_distribute = general_bbox.width * 2 > general_bbox.height
        else:
            horizontal_distribute = self.axis == 'X'

        update_tag = False
        cursor_offset = 0
        if horizontal_distribute:
            _islands.sort(key=lambda a: a.bbox.xmin)

            general_bbox.xmax += self.padding * (len(_islands) - 1)
            start_space = general_bbox.xmin + _islands[0].bbox.half_width
            end_space = general_bbox.xmax - _islands[-1].bbox.half_width
            if start_space == end_space:
                umeshes.cancel_with_report(info=f"No distance to place UV")
                return

            if cursor:
                diff = end_space - start_space
                start_space += cursor.x - start_space
                end_space = start_space + diff
                cursor_offset += general_bbox.ymin - cursor.y
            space_points = np.linspace(start_space, end_space, len(_islands))

            for island, space_point in zip(_islands, space_points):
                update_tag |= island.set_position(Vector((space_point, island.bbox.center_y - cursor_offset)), _from=island.bbox.center)
        else:
            _islands.sort(key=lambda a: a.bbox.ymin)
            general_bbox.ymax += self.padding * (len(_islands) - 1)
            start_space = general_bbox.ymin + _islands[0].bbox.half_height
            end_space = general_bbox.ymax - _islands[-1].bbox.half_height
            if start_space == end_space:
                umeshes.cancel_with_report(info=f"No distance to place UV")
                return
            if cursor:
                start_space += start_space - cursor.y
                end_space += end_space - cursor.y

            if cursor:
                diff = end_space - start_space
                start_space += cursor.y - start_space
                end_space = start_space + diff
                cursor_offset += general_bbox.xmin - cursor.x

            space_points = np.linspace(start_space, end_space, len(_islands))

            for island, space_point in zip(_islands, space_points):
                update_tag |= island.set_position(Vector((island.bbox.center_x - cursor_offset, space_point)), _from=island.bbox.center)

        if not update_tag:
            umeshes.cancel_with_report(info='Islands is Distributed')

    @staticmethod
    def distribute_preprocessing(sync, umeshes, extended):
        _islands: list[AdvIsland] = []
        general_bbox = BBox()
        for umesh in umeshes:
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                general_bbox.union(adv_islands.calc_bbox())
                _islands.extend(adv_islands)
            umesh.update_tag = bool(adv_islands)
        return _islands, general_bbox

    @staticmethod
    def distribute_preprocessing_overlap(sync, umeshes, extended):
        _islands: list[AdvIsland] = []
        for umesh in umeshes:
            if adv_islands := AdvIslands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                adv_islands.calc_tris()
                adv_islands.calc_flat_coords()
                _islands.extend(adv_islands)
            umesh.update_tag = bool(adv_islands)

        general_bbox = BBox()
        union_islands_groups = UnionIslands.calc_overlapped_island_groups(_islands)
        for union_island in union_islands_groups:
            general_bbox.union(union_island.bbox)
        return union_islands_groups, general_bbox

class UNIV_OT_Home(Operator):
    bl_idname = 'uv.univ_home'
    bl_label = 'Home'
    bl_description = 'Home'
    bl_options = {'REGISTER', 'UNDO'}

    mode: EnumProperty(name='Mode', default='DEFAULT', items=(
        ('DEFAULT', 'Default', ''),
        ('TO_CURSOR', 'To Cursor', ''),
    ))

    # ('OVERLAPPED', 'Overlapped', '')

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        match event.ctrl, event.shift, event.alt:
            case False, False, False:
                self.mode = 'DEFAULT'
            case True, False, False:
                self.mode = 'TO_CURSOR'
            # case False, True, False:
            #     self.mode = 'OVERLAPPED'
            case _:
                self.report({'INFO'}, f"Event: {info.event_to_string(event)} not implement.\n\n"
                                      f"See all variations:\n\n")
                return {'CANCELLED'}
        return self.execute(context)

    def execute(self, context):
        return UNIV_OT_Home.home(self.mode, sync=context.scene.tool_settings.use_uv_select_sync, report=self.report)

    @staticmethod
    def home(mode, sync, report):
        umeshes = utils.UMeshes(report=report)
        match mode:
            case 'DEFAULT':
                UNIV_OT_Home.home_ex(umeshes, sync, extended=True)
                if not umeshes.final():
                    UNIV_OT_Home.home_ex(umeshes, sync, extended=False)

            case 'TO_CURSOR':
                if not (cursor_loc := utils.get_tile_from_cursor()):
                    umeshes.report({'WARNING'}, "Cursor not found")
                    return {'CANCELLED'}
                UNIV_OT_Home.home_ex(umeshes, sync, extended=True, cursor=cursor_loc)
                if not umeshes.final():
                    UNIV_OT_Home.home_ex(umeshes, sync, extended=False, cursor=cursor_loc)

            case _:
                raise NotImplementedError(mode)

        return umeshes.update()

    @staticmethod
    def home_ex(umeshes, sync, extended, cursor=Vector((0, 0))):
        for umesh in umeshes:
            changed = False
            if islands := Islands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, sync, extended=extended):
                for island in islands:
                    center = island.calc_bbox().center
                    delta = Vector(round(-i + 0.5) for i in center) + cursor
                    changed |= island.move(delta)
            umesh.update_tag = changed


class UNIV_OT_Random(Operator):
    bl_idname = "uv.univ_random"
    bl_label = "Random"
    bl_description = "Randomize selected UV islands or faces"
    bl_options = {'REGISTER', 'UNDO'}

    overlapped: BoolProperty(name='Overlapped', default=False)
    between: BoolProperty(name='Between', default=False)
    bound_between: EnumProperty(name='Bound Between', default='OFF', items=(('OFF', 'Off', ''), ('CROP', 'Crop', ''), ('CLAMP', 'Clamp', '')))
    round_mode: EnumProperty(name='Round Mode', default='OFF', items=(('OFF', 'Off', ''), ('INT', 'Int', ''), ('STEPS', 'Steps', '')))
    steps: FloatVectorProperty(name='Steps', description="Incorrectly works with Within Image Bounds",
                               default=(0, 0), min=0, max=10, soft_min=0, soft_max=1, size=2, subtype='XYZ')
    strength: FloatVectorProperty(name='Strength', default=(1, 1), min=-10, max=10, soft_min=0, soft_max=1, size=2, subtype='XYZ')
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
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

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

        if not self.between:
            layout.prop(self, 'bool_bounds')

        layout = self.layout.row()
        if self.between:
            layout.prop(self, 'bound_between', expand=True)
        layout = self.layout.row()
        layout.prop(self, 'overlapped', toggle=1)
        layout.prop(self, 'between', toggle=1)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)
        self.overlapped = event.shift
        self.between = event.alt
        self.bound_between = 'CROP' if event.ctrl else 'OFF'
        return self.execute(context)

    def __init__(self):
        self.seed = 1000
        self.non_valid_counter = 0
        self.umeshes: utils.UMeshes | None = None
        self.all_islands: list[UnionIslands | AdvIsland] | None = None
        self.sync = bpy.context.scene.tool_settings.use_uv_select_sync

    def execute(self, context):
        self.non_valid_counter = 0
        self.umeshes = utils.UMeshes(report=self.report)
        self.random_preprocessing()

        if not self.all_islands:
            self.random_preprocessing(extended=False)

        if self.between:
            self.random_between()
        else:
            self.random()
        if self.non_valid_counter:
            self.report({'INFO'}, f"Found {self.non_valid_counter} zero-sized islands that will not be affected by some effects")
        return self.umeshes.update(info="No object for randomize.")

    def random_preprocessing(self, extended=True):
        self.seed = sum(id(umesh.obj) for umesh in self.umeshes) // len(self.umeshes) + self.rand_seed

        self.all_islands = []
        _islands = []
        for umesh in self.umeshes:
            if islands := AdvIslands.calc_extended_or_visible(umesh.bm, umesh.uv_layer, self.sync, extended=extended):
                if self.overlapped:
                    islands.calc_tris()
                    islands.calc_flat_coords()
                    _islands.extend(islands)
                else:
                    self.all_islands.extend(islands)
            umesh.update_tag = bool(islands)

        if self.overlapped:
            self.all_islands = UnionIslands.calc_overlapped_island_groups(_islands)

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

                    if island.rotate(angle, vec_origin):
                        bb.rotate_expand(angle)

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

                    island.rotate(angle, vec_origin)
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

            if self.overlapped:
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
