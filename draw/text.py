# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import blf
import gpu
import typing

from time import perf_counter as time

from .. import utils


class TextDraw:
    """NOTE: max_draw_time and target_area automatically revert to their default values."""
    start_time = time()
    max_draw_time = 1.5
    width: int = 0
    height: int = 0
    y_pad__with_text: list[tuple[int, str]] = []
    handler: None = None
    target_area: typing.Literal['UV', 'VIEW_3D'] = 'UV'

    @classmethod
    def draw(cls, text: str | list[str]):
        cls.start_time = time()
        cls._text_precessing(text)

        if cls.target_area == 'UV':
            sima_or_view3d = bpy.types.SpaceImageEditor
        else:
            sima_or_view3d = bpy.types.SpaceView3D
        if not (cls.handler is None):
            sima_or_view3d.draw_handler_remove(cls.handler, 'WINDOW')

        cls.handler = sima_or_view3d.draw_handler_add(cls.draw_callback_px_uv_area, (), 'WINDOW', 'POST_PIXEL')
        bpy.app.timers.register(cls.uv_area_draw_timer)

    @classmethod
    def uv_area_draw_timer(cls):
        if cls.handler is None:
            cls.max_draw_time = 1.5
            cls.target_area = 'UV'
            return
        counter = time() - cls.start_time

        if counter < cls.max_draw_time:
            return 0.2

        if cls.target_area == 'UV':
            sima_or_view3d = bpy.types.SpaceImageEditor
        else:
            sima_or_view3d = bpy.types.SpaceView3D
        sima_or_view3d.draw_handler_remove(cls.handler, 'WINDOW')

        for a in bpy.context.screen.areas:
            if cls.target_area == 'UV':
                if a.type == 'IMAGE_EDITOR' and a.ui_type == 'UV':
                    a.tag_redraw()
            elif cls.target_area == 'VIEW_3D':
                if a.type == 'VIEW_3D':
                    a.tag_redraw()

        cls.handler = None
        cls.max_draw_time = 1.5
        cls.target_area = 'UV'
        return

    @classmethod
    def draw_callback_px_uv_area(cls):
        area = bpy.context.area
        if cls.target_area == 'UV':
            if area.ui_type != 'UV':
                return
        else:
            if area.type != cls.target_area:
                return

        n_panel_width = next(r.width for r in area.regions if r.type == 'UI')
        if (area.width - n_panel_width) < cls.width or area.height < cls.height:
            return

        x_pos = area.width - n_panel_width - cls.width

        gpu.state.blend_set('ALPHA')
        font_id = 0
        utils.blf_size(font_id, 16)

        blf.color(font_id, 0.95, 0.95, 0.95, 0.85)
        for y_pad, txt in cls.y_pad__with_text:
            blf.position(font_id, x_pos, y_pad, 0)
            blf.draw(font_id, txt)

        gpu.state.blend_set('NONE')

    @classmethod
    def _text_precessing(cls, text):
        text = [text] if isinstance(text, str) else text
        text_x_size, char_y_size = blf.dimensions(0, 'T')
        pad_x = 40
        cls.width = max(len(line) for line in text) * text_x_size + pad_x

        pad_y = 20
        cls.height = pad_y * (len(text) - 1) + char_y_size * len(text) + pad_y

        text_y_size = cls.height
        y_pad__with_text = []
        for txt in text:
            y_pad__with_text.append((text_y_size, txt))
            text_y_size -= char_y_size + pad_y
        cls.y_pad__with_text = y_pad__with_text
