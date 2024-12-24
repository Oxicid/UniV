# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later
import os
class icons:
    _icons_ = None
    adjust = 0
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
    center = 0
    checker = 0
    crop = 0
    cursor = 0
    cut = 0
    distribute = 0
    edge_grow = 0
    fill = 0
    flip = 0
    flipped = 0
    grow = 0
    home = 0
    horizontal_a = 0
    horizontal_c = 0
    large = 0
    medium = 0
    non_splitted = 0
    normalize = 0
    orient = 0
    overlap = 0
    pack = 0
    pin = 0
    quadrify = 0
    random = 0
    relax = 0
    remove = 0
    rotate = 0
    settings_a = 0
    settings_b = 0
    shift = 0
    small = 0
    sort = 0
    square = 0
    stack = 0
    stitch = 0
    straight = 0
    unwrap = 0
    vertical_a = 0
    vertical_b = 0
    weld = 0
    x = 0
    y = 0
    zero = 0

    @classmethod
    def register_icons_(cls):
        from bpy.utils import previews
        if cls._icons_ is None:
            cls._icons_ = previews.new()
        else:
            cls.reset_icon_value_()
        png_file_path = __file__.replace('__init__.py', 'png/')

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
                setattr(cls, attr, icon.icon_id)

    @classmethod
    def reset_icon_value_(cls):
        for attr in dir(cls):
            if not attr.endswith('_'):
                setattr(cls, attr, 0)

    @classmethod
    def unregister_icons_(cls):
        from bpy.utils import previews
        previews.remove(cls._icons_)
        cls.reset_icon_value_()

    @staticmethod
    def install_dependencies_():
        from ..utils import Pip
        Pip.install('rl-renderPM')
        Pip.install('pycairo')
        Pip.install('reportlab[renderPM]')
        Pip.install('reportlab')
        Pip.install('libpng')  # noqa

    @classmethod
    def convert_svg_to_png_(cls, texture_size=32):
        try:
            import fitz
            from svglib import svglib
            from reportlab.graphics import renderPDF
        except ImportError:
            print('UniV: No install dependencies (fitz, svglib or reportlab)')
            return

        svg_file_path = __file__.replace('__init__.py', 'svg/').replace('\\', '/')
        png_file_path = __file__.replace('__init__.py', 'png/').replace('\\', '/')

        for attr in dir(cls):
            if not attr.endswith('_'):
                assert isinstance(getattr(cls, attr), int)

                svg_file = svg_file_path + 'univ_icon_' + attr + ".svg"
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
