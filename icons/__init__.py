# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

# The icons were created by Vitaly Zhdanov https://www.youtube.com/@diffusecolor , for which he is very thankful!
# His work gave the project a finished and professional look.
# Excellent detailing and stylish design made the interface more convenient and pleasant.
# Thank you for the work done!

import os


class icons:
    _icons_ = None
    adjust = 0
    arrow = 0
    arrow_bottom = 0
    arrow_bottom_left = 0
    arrow_bottom_right = 0
    arrow_left = 0
    arrow_right = 0
    arrow_top = 0
    arrow_top_left = 0
    arrow_top_right = 0
    border = 0
    border_by_angle = 0
    border_seam = 0
    box = 0
    center = 0
    checker = 0
    crop = 0
    cursor = 0
    cut = 0
    distribute = 0
    edge_grow = 0
    fill = 0
    flat = 0
    flip = 0
    flipped = 0
    gravity = 0
    grow = 0
    home = 0
    horizontal_a = 0
    horizontal_c = 0
    large = 0
    linked = 0
    loop_select = 0
    medium = 0
    non_splitted = 0
    normal = 0
    normalize = 0
    orient = 0
    overlap = 0
    pack = 0
    pin = 0
    quadrify = 0
    random = 0
    rectify = 0
    relax = 0
    remove = 0
    reset = 0
    rotate = 0
    select_stacked = 0
    settings_a = 0
    settings_b = 0
    shift = 0
    small = 0
    smart = 0
    sort = 0
    square = 0
    stack = 0
    stitch = 0
    straight = 0
    transfer = 0
    unwrap = 0
    vertical_a = 0
    vertical_b = 0
    view = 0
    weld = 0
    x = 0
    y = 0
    zero = 0

    @classmethod
    def register_icons_(cls):
        from bpy.utils import previews
        if cls._icons_:
            cls.unregister_icons_()
        cls._icons_ = previews.new()

        from ..preferences import prefs
        png_folder_name = 'png_mono/' if prefs().color_mode == 'MONO' else 'png/'
        png_file_path = __file__.replace('__init__.py', png_folder_name)

        for attr in dir(cls):
            if not attr.endswith('_'):
                if not isinstance(getattr(cls, attr), int):
                    print(f'UniV: Attribute {attr} not icon_id')
                    continue

                full_path = png_file_path + attr + ".png"
                if not os.path.exists(full_path):
                    print(f'UniV: File {full_path} for {attr} icon not found')
                    continue

                icon = cls._icons_.load(attr, full_path, 'IMAGE')
                _ = icon.icon_pixels[0]  # Need to force load icons, bugreport?
                setattr(cls, attr, icon.icon_id)

        import bpy
        from pathlib import Path
        from .. import ui

        panels = [ui.UNIV_WT_edit_VIEW3D, ui.UNIV_WT_object_VIEW3D]

        icon_path = Path(panels[0].bl_icon)
        expected_icon = 'univ_mono' if prefs().color_mode == 'MONO' else 'univ'

        if icon_path.parts[-1] != expected_icon:
            new_path = icon_path.parent / expected_icon
            for p in panels:
                try:
                    bpy.utils.unregister_tool(p)
                    p.bl_icon = str(new_path)
                    bpy.utils.register_tool(p)
                except Exception as e:
                    print(f'UniV: Updating icons for workspaces has failed:\n{e}')

    @classmethod
    def reset_icon_value_(cls):
        for attr in dir(cls):
            if not attr.endswith('_'):
                setattr(cls, attr, 0)

    @classmethod
    def unregister_icons_(cls):
        from bpy.utils import previews
        try:
            previews.remove(cls._icons_)
        except KeyError:
            from ..preferences import debug
            if debug():
                import traceback
                traceback.print_exc()

        cls.reset_icon_value_()

    @staticmethod
    def install_dependencies_():
        from ..utils import Pip
        Pip.install('rl-renderPM')
        Pip.install('pycairo')
        Pip.install('reportlab[renderPM]')
        Pip.install('reportlab')
        Pip.install('libpng')  # noqa
        Pip.install('pymupdf')  # noqa
        Pip.install('svglib')  # noqa
        Pip.install('frontend')  # noqa

    @classmethod
    def convert_svg_to_png_(cls, texture_size=32, mono=False):
        try:
            import fitz
            from svglib import svglib
            from reportlab.graphics import renderPDF
        except ImportError:
            print('UniV: No install dependencies (fitz, svglib or reportlab)')
            return

        svg_folder_name = 'svg/'
        png_folder_name = 'png/'
        if mono:
            svg_folder_name = 'svg_mono/'
            png_folder_name = 'png_mono/'
        svg_file_path = __file__.replace('__init__.py', svg_folder_name).replace('\\', '/')
        png_file_path = __file__.replace('__init__.py', png_folder_name).replace('\\', '/')

        for attr in dir(cls):
            if not attr.endswith('_'):
                assert isinstance(getattr(cls, attr), int)

                svg_file = svg_file_path + attr + ".svg"
                if not os.path.exists(svg_file):
                    print(f'UniV: File {svg_file} not found')
                    continue
                png_save_path = png_file_path + attr + ".png"

                # Convert svg to pdf in memory with svglib+reportlab
                # directly rendering to png does not support transparency nor scaling
                drawing = svglib.svg2rlg(path=svg_file)
                pdf = renderPDF.drawToString(drawing)

                # Open pdf with fitz (pyMuPdf) to convert to PNG
                doc = fitz.Document(stream=pdf)

                width_in_inches = doc[0].rect.width / 72

                dpi = texture_size / width_in_inches
                pix = doc.load_page(0).get_pixmap(alpha=True, dpi=round(dpi))
                pix.save(png_save_path)
