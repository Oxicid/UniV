# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy  # noqa

import gpu
import typing
import numpy as np
from .. import utils
from mathutils import Color


class UTexture:
    def __init__(self, w, h, texture=None, channels: int=3):
        self.width = w
        self.height = h

        if texture is None:
            self.data = np.zeros(shape=(h, w, channels), dtype=np.float32)
            self.channels = channels
        else:
            self.data = texture
            self.channels = texture.shape[2]

    @classmethod
    def from_ibuf(cls, ibuf):
        from ..import utypes
        width, height = ibuf.size

        c_ibuf = utypes.Py_ImBuf.get_fields(ibuf)
        data = np.ctypeslib.as_array(c_ibuf.byte_buffer.data, shape=[width, height, 4])
        data = data / np.float32(255)

        assert ibuf.channels == 4
        return cls(width, height, data, channels=ibuf.channels)

    @classmethod
    def from_frame_buf(cls, width, height, fb):
        pixel_data = fb.read_color(0, 0, width, height, 4, 0, 'UBYTE')
        pixel_data.dimensions = width * height * 4
        t = np.array(pixel_data, dtype=np.uint8) / np.float32(255)
        t = t.reshape([width, height, 4])
        return cls(width, height, t, channels=4)

    def to_gpu_texture(self):
        texture_with_alpha = self.data
        if self.channels == 3:
            alpha = np.ones((self.height, self.width, 1), dtype=self.data.dtype)
            texture_with_alpha = np.concatenate((self.data, alpha), axis=2)

        assert texture_with_alpha.dtype == np.float32
        buffer = gpu.types.Buffer('FLOAT', self.width * self.height * 4, texture_with_alpha)
        return gpu.types.GPUTexture((self.width, self.height), format='RGBA8', data=buffer)

    def fill(self, color):
        self.data[:] = self._sanitize_color(color)

    def _sanitize_color(self, col):
        col = np.asarray(col, dtype=np.float32)

        if len(col) != self.channels:
            if len(col) == 3:
                return np.concatenate((col, [1.0]))
            else:
                return col[:3]
        else:
            return col

    def lerp(self, other: 'typing.Self', factor):
        diff = other.data - self.data
        diff *= factor
        diff += self.data
        return UTexture(self.width, self.height, diff)

    def offset(self, x: int, y: int):
        self.data = np.roll(self.data, shift=(x, y), axis=(0, 1))


    def __add__(self, other: 'typing.Self'):
        if isinstance(other, UTexture):
            other_tex = other.data
        else:
            other_tex = np.float32(other)

        texture = self.data.copy()

        if isinstance(other_tex, np.ndarray):
            texture[..., :3] += other_tex[..., :3]
        else:
            texture[..., :3] += other_tex

        return UTexture(self.width, self.height, texture)

    def __iadd__(self, other: 'typing.Self'):
        if isinstance(other, UTexture):
            other = other.data
            self.data[..., :3] += other[..., :3]
        else:
            self.data[..., :3] += other

        return self

    def __sub__(self, other: 'typing.Self'):
        if isinstance(other, UTexture):
            other_tex = other.data
        else:
            other_tex = np.float32(other)

        texture = self.data.copy()

        if isinstance(other_tex, np.ndarray):
            texture[..., :3] -= other_tex[..., :3]
        else:
            texture[..., :3] -= other_tex

        return UTexture(self.width, self.height, texture)

    def __isub__(self, other: 'typing.Self'):
        if isinstance(other, UTexture):
            other = other.data

        self.data[..., :3] -= other
        return self

    def __mul__(self, other: 'typing.Self'):
            if isinstance(other, UTexture):
                other_tex = other.data
            else:
                other_tex = np.float32(other)

            texture = self.data.copy()

            if isinstance(other_tex, np.ndarray):
                texture[..., :3] *= other_tex[..., :3]
            else:
                texture[..., :3] *= other_tex

            return UTexture(self.width, self.height, texture)

    def __imul__(self, other: 'typing.Self'):
        if isinstance(other, UTexture):
            other = other.data

        self.data[..., :3] *= other
        return self


    def __setitem__(self, key, value):
        if isinstance(key, UMask):
            self.data[key.data] = value
        else:
            self.data[key] = value



class UMask:
    def __init__(self, w, h, texture=None):
        self.width = w
        self.height = h

        if texture is None:
            self.data = np.zeros(shape=(h, w), dtype=bool)
        else:
            self.data = texture

    # def to_gpu_texture(self):
    #     raise

    def offset(self, x: int, y: int):
        self.data = np.roll(self.data, shift=(x, y), axis=(0, 1))

    def mask_to_texture(self, col_a, col_b):
        assert len(col_a) == len(col_b)
        col_a = np.asarray(col_a, dtype=np.float32)
        col_b = np.asarray(col_b, dtype=np.float32)

        texture = np.where(self.data[..., None], col_a, col_b)
        return UTexture(self.width, self.height, texture, channels=len(col_b))

    def __add__(self, other: 'typing.Self'):
        assert type(other) is UMask
        tex = self.data | other.data
        return UMask(self.width, self.height, tex)

    def __iadd__(self, other: 'typing.Self'):
        assert type(other) is UMask
        self.data |= other.data
        return self

    def __sub__(self, other: 'typing.Self'):
        assert type(other) is UMask
        tex = self.data & ~other.data
        return UMask(self.width, self.height, tex)

    def __isub__(self, other: 'typing.Self'):
        assert type(other) is UMask
        self.data &= ~other.data
        return self

    def __ior__(self, other: 'typing.Self'):
        """Union masks."""
        assert type(other) is UMask
        self.data |= other.data
        return self

    def __or__(self, other: 'typing.Self'):
        """Union masks."""
        assert type(other) is UMask

        texture = self.data | other.data
        return UMask(self.width, self.height, texture)

    def __iand__(self, other: 'typing.Self'):
        """Inplace isect masks."""
        assert type(other) is UMask
        self.data &= other.data
        return self

    def __and__(self, other: 'typing.Self'):
        """Isect masks."""
        assert type(other) is UMask

        texture = self.data & other.data
        return UMask(self.width, self.height, texture)

    def __ixor__(self, other: 'typing.Self'):
        """Toggle masks."""
        assert type(other) is UMask
        self.data ^= other.data
        return self

    def __xor__(self, other: 'typing.Self'):
        """Toggle masks."""
        assert type(other) is UMask

        texture = self.data ^ other.data
        return UMask(self.width, self.height, texture)

    def __setitem__(self, key, value):
        if isinstance(key, UTexture):
            self.data[key.data] = value
        else:
            self.data[key] = value

    def draw_rect(self, y0: int, y1: int, x0: int, x1: int, value: bool=True):
        """ Fill wrapped rect mask. """

        for y_slice in self.wrap_slices_h(y0, y1):
            for x_slice in self.wrap_slices_w(x0, x1):
                self.data[y_slice, x_slice] = value

    def wrap_slices_w(self, start, stop):
        """ Returns wrapped slices for width. """
        length = stop - start
        s0 = start % self.width
        s1 = s0 + length

        if s1 <= self.width:
            return [slice(s0, s1)]
        else:
            return [slice(s0, self.width), slice(0, s1 - self.width)]

    def wrap_slices_h(self, start, stop):
        """ Returns wrapped slices for height. """
        length = stop - start
        s0 = start % self.height
        s1 = s0 + length

        if s1 <= self.height:
            return [slice(s0, s1)]
        else:
            return [slice(s0, self.height), slice(0, s1 - self.height)]

    def draw_vline(self, x, thickness, value=True):
        """ Draw wrapped vertical mask line. """
        half = thickness // 2

        x0 = x - half
        x1 = x + half + (thickness & 1)

        for xs in self.wrap_slices_w(x0, x1):
            self.data[:, xs] = value

    def draw_hline(self, y, thickness, value=True):
        """ Draw wrapped horizontal mask line. """
        half = thickness // 2

        y0 = y - half
        y1 = y + half + (thickness & 1)

        for ys in self.wrap_slices_h(y0, y1):
            self.data[ys, :] = value


class Colors:
    gray = (0.25, 0.25, 0.25)  # good
    black = 0.005, 0.005, 0.005  # good

    _separator1 = None

    brown = 0.256, 0.092, 0.0  # good
    clay = 0.505, 0.325, 0.156  # good, but small lines gray (need light)

    _separator2 = None

    yellow = (0.92, 0.8, 0.11)  # good, bot need darken lines
    yellow_green = (0.715,0.89,0.0)  # good, bot need darken big lines

    _separator3 = None

    red = (0.576, 0.058, 0.027)  # good, but small lines gray (need light)
    red_orange = [1, 0.18, 0]  # good
    coral = [1.0, 0.4, 0.31]  # good, but small lines gray (need light)

    _separator4 = None

    orange = (1.0, 0.647, 0.0)  # good
    orange_other = (1,0.33,0)  # good

    _separator5 = None

    green = (0.337, 0.51, 0.0115)  # good, but need colorize lines
    green_darken = (0.08,0.43,0.08)
    army_green = [0.33, 0.38, 0.12]  # good
    hunter_green = (0.172, 0.372, 0.204)  # good

    _separator6 = None

    blue = [0.0, 0.18, 0.58]  # good
    midnight_blue = [0.08, 0.13, 0.37]  # good
    blueprint = (0.188,0.36,0.87)  # good, but need colorize small lines
    azure = [0.35, 0.98, 0.95]  # need colorize lines
    aquamarine = [0.0, 1.0, 0.74]

    _separator7 = None

    violet = [0.47, 0.3, 0.5]  # good, but lines - not
    deep_violet = [0.22, 0.04, 0.4]  # good

    _separator8 = None

    pink = [0.7, 0.0, 0.22]  # good

class ColorsSmallLines:
    gray = (0.35,0.35,0.35)
    black = (0.35,0.35,0.35)

    brown = Color(Colors.brown)  # good, but small lines gray (need light)
    brown.s *= 0.6
    brown.v *= 2.0

    clay = Color(Colors.clay)  # good, but small lines gray (need light)
    clay.s *= 1.5
    clay.v *= 0.7

    _separator2 = None

    yellow = Color(Colors.yellow)
    yellow.v *= 0.5
    yellow.s *= 1.5

    yellow_green = Color(Colors.yellow_green)
    yellow_green.v *= 0.5
    yellow_green.s *= 1.5



    _separator3 = None

    orange = Color(Colors.orange)
    orange.v *= 0.65
    orange.s *= 1.5

    orange_other = Color(Colors.orange_other)
    orange_other.v *= 0.65
    orange_other.s *= 1.5


    red_orange = Color(Colors.red_orange)
    red_orange.v *= 0.65
    red_orange.s *= 1.5

    _separator4 = None

    red = Color(Colors.red)
    red.v *= 1.55
    red.s *= 0.65


    coral = Color(Colors.coral)
    coral.v *= 0.65
    coral.s *= 1.5


    green = Color(Colors.green)
    green.v *= 0.65
    green.s *= 1.5

    green_darken = Color(Colors.green_darken)
    green_darken.v *= 0.65
    green_darken.s *= 1.5

    army_green = Color(Colors.army_green)
    army_green.v *= 0.65
    army_green.s *= 1.5


    hunter_green = Color(Colors.hunter_green)
    hunter_green.v *= 0.65
    hunter_green.s *= 1.5

    blue = Color(Colors.blue)
    blue.v *= 1.75
    blue.s *= 0.55

    blueprint = Color(Colors.blueprint)
    blueprint.v *= 0.65
    blueprint.s *= 1.5

    midnight_blue = Color(Colors.midnight_blue)
    midnight_blue.v *= 1.75
    midnight_blue.s *= 0.75




    aquamarine = Color(Colors.aquamarine)
    aquamarine.v *= 0.65
    aquamarine.s *= 1.5

    azure = Color(Colors.azure)
    azure.v *= 0.65
    azure.s *= 1.5

    violet = Color(Colors.violet)
    violet.v *= 0.85
    violet.s *= 2.0

    deep_violet = Color(Colors.deep_violet)
    deep_violet.v *= 1.65
    deep_violet.s *= 0.8

    pink = Color(Colors.pink)
    pink.v *= 1.65
    pink.s *= 0.8



class TexturePatterns:
    @staticmethod
    def draw_lines(tex: UMask, step=32, thickness=1, exclude_first=False):
        first = step if exclude_first else 0
        for x in range(first, tex.width, step):
            tex.draw_vline(x, thickness)

        for y in range(first, tex.height, step):
            tex.draw_hline(y, thickness)

    @staticmethod
    def draw_dashed_hline(tex: UMask, y, dash=6, gap=4, thickness=1, phase=0):
        x = np.arange(tex.width)
        mask = ((x + phase) % (dash + gap)) < dash

        half = thickness // 2
        y0 = y - half
        y1 = y + half + (thickness & 1)

        for ys in tex.wrap_slices_h(y0, y1):
            tex.data[ys, mask] = True


    @staticmethod
    def draw_dashed_vline(tex: UMask, x, dash=6, gap=4, thickness=1, phase=0):
        y = np.arange(tex.height)
        mask = ((y + phase) % (dash + gap)) < dash

        half = thickness // 2
        x0 = x - half
        x1 = x + half + (thickness & 1)

        for xs in tex.wrap_slices_w(x0, x1):
            tex.data[mask, xs] = True

    @classmethod
    def draw_pluses(cls, texture: UMask, step: int=128, size: int=32, thickness: int=3, center=True):
        for x, y in utils.grid_points_px(texture.width, texture.height, step, center):
            cls.draw_plus(texture, y, x, size=size, thickness=thickness)

    @staticmethod
    def draw_plus(tex: UMask, cy: int, cx: int, size: int, thickness: int):
        half_size = size // 2
        half_th = thickness // 2

        # Vertical
        tex.draw_rect(
            cy - half_size,
            cy + half_size,
            cx - half_th,
            cx + half_th + 1
        )

        if size == thickness:
            return

        # Horizontal
        tex.draw_rect(
            cy - half_th,
            cy + half_th + 1,
            cx - half_size,
            cx + half_size
        )


    @classmethod
    def draw_checker(cls, mask, step=32):
        y = np.arange(mask.height)[:, None]
        x = np.arange(mask.width)[None, :]

        mask.data[:] = ((x // step + y // step) & 1) == 0


    @staticmethod
    def checker_board_text(width: int, height: int, step: int = 128, outline: int = 1):
        import blf
        mono: int = 0 #  blf_mono_font_render;
        text_size = 54  # hard coded size!
        utils.blf_size(mono, text_size)

        # Using nullptr will assume the byte buffer has sRGB colorspace, which currently
        # matches the default colorspace of new images.

        text_color: list[float] = [0.0, 0.0, 0.0, 1.0]
        text_outline: list[float] = [1.0, 1.0, 1.0, 1.0]

        import string
        letters = string.ascii_uppercase
        digits = string.digits[1:] + 'ABCDEF'

        first_char_index: int = 0
        for y in range(0, height, step):
            first_char = letters[first_char_index]

            second_char_index: int = 0
            for x in range(0, width, step):
                second_char = digits[second_char_index]
                text = first_char + second_char
                size_x, size_y = blf.dimensions(mono, text)

                # hard coded offset
                pen_x: int = x + ((step // 2) - (size_x // 2))
                pen_y: int = y + ((step // 2) - (size_y // 2))

                # terribly crappy outline font!
                blf.color(mono, *text_outline)

                for dx, dy in utils.padding_deltas(outline):

                    blf.position(mono, pen_x+dx, pen_y+dy, 0.0)
                    blf.draw_buffer(mono, text)

                blf.color(mono, *text_color)
                blf.position(mono, pen_x, pen_y, 0.0)
                blf.draw_buffer(mono, text)

                second_char_index = (second_char_index + 1) % len(digits)

            first_char_index = (first_char_index + 1) % len(letters)

    @classmethod
    def simple_line(cls, size=(2048, 2048), color=(0.25, 0.25, 0.25), small_lines_color=(0.4,0.4,0.4)):
        bound_line = UMask(*size)
        bound_line.draw_vline(0, thickness=5)
        bound_line.draw_hline(0, thickness=5)

        medium_lines = UMask(*size)
        cls.draw_lines(medium_lines, step=256, exclude_first=True)

        small_lines = UMask(*size)
        cls.draw_lines(small_lines, step=32, exclude_first=True)

        tex = small_lines.mask_to_texture(small_lines_color, color)
        tex[medium_lines] = (1,1,1)
        tex[bound_line] = (1,1,1)
        return tex