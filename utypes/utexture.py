import gpu
import typing
import numpy as np


class UTexture:
    def __init__(self, w, h, texture=None, texture_type=float):
        self.width = w
        self.height = h
        self.texture_type = texture_type

        if texture is None:
            if texture_type is bool:
                self.texture = np.zeros(shape=(h, w), dtype=bool)
            else:
                self.texture = np.zeros(shape=(h, w, 4), dtype=np.float32)
        else:
            self.texture = texture

    @classmethod
    def from_ibuf(cls, ibuf):
        from ..import utypes
        width, height = ibuf.size

        c_ibuf = utypes.Py_ImBuf.get_fields(ibuf)
        data = np.ctypeslib.as_array(c_ibuf.byte_buffer.data, shape=[width, height, 4])
        data = data / np.float32(255)

        assert ibuf.channels == 4
        return cls(width, height, data)

    @classmethod
    def from_frame_buf(cls, width, height, fb):
        pixel_data = fb.read_color(0, 0, width, height, 4, 0, 'UBYTE')
        pixel_data.dimensions = width * height * 4
        t = np.array(pixel_data, dtype=np.uint8) / np.float32(255)
        return cls(width, height, t)

    def to_gpu_texture(self):
        assert self.texture_type is float
        buffer = gpu.types.Buffer('FLOAT', self.width * self.height * 4, self.texture)
        return gpu.types.GPUTexture((self.width, self.height), format='RGBA8', data=buffer)

    def fill(self, color):
        self.texture[:] = color

    def lerp(self, other: 'typing.Self', factor):
        diff = other.texture - self.texture
        diff *= factor
        diff += self.texture
        return UTexture(self.width, self.height, diff)

    def offset(self, x: int, y: int):
        self.texture = np.roll(self.texture, shift=(x, y), axis=(0, 1))

    def mask_to_texture(self, col_a, col_b):
        assert self.texture_type is bool
        col_a = np.asarray(col_a, dtype=np.float32)
        col_b = np.asarray(col_b, dtype=np.float32)

        texture = np.where(self.texture[..., None], col_b, col_a)
        return UTexture(self.width, self.height, texture)

    def __add__(self, other: 'typing.Self'):
        if isinstance(other, UTexture):
            other_tex = other.texture
        else:
            other_tex = np.float32(other)

        texture = self.texture.copy()

        if isinstance(other_tex, np.ndarray):
            texture[..., :3] += other_tex[..., :3]
        else:
            texture[..., :3] += other_tex

        return UTexture(self.width, self.height, texture, texture_type=float)

    def __iadd__(self, other: 'typing.Self'):
        if isinstance(other, UTexture):
            other = other.texture
            self.texture[..., :3] += other[..., :3]
        else:
            self.texture[..., :3] += other

        return self

    def __sub__(self, other: 'typing.Self'):
        if self.texture_type is bool:
            assert other.texture_type is bool
            texture = self.texture & ~other.texture
            return UTexture(self.width, self.height, texture, texture_type=bool)
        else:
            if isinstance(other, UTexture):
                other_tex = other.texture
            else:
                other_tex = np.float32(other)

            texture = self.texture.copy()

            if isinstance(other_tex, np.ndarray):
                texture[..., :3] -= other_tex[..., :3]
            else:
                texture[..., :3] -= other_tex

            return UTexture(self.width, self.height, texture, texture_type=float)

    def __isub__(self, other: 'typing.Self'):
        if self.texture_type is bool:
            assert other.texture_type is bool
            self.texture &= ~other.texture
        else:
            if isinstance(other, UTexture):
                other = other.texture

            self.texture[..., :3] -= other
        return self

    def __mul__(self, other: 'typing.Self'):
            if isinstance(other, UTexture):
                other_tex = other.texture
            else:
                other_tex = np.float32(other)

            texture = self.texture.copy()

            if isinstance(other_tex, np.ndarray):
                texture[..., :3] *= other_tex[..., :3]
            else:
                texture[..., :3] *= other_tex

            return UTexture(self.width, self.height, texture, texture_type=float)

    def __imul__(self, other: 'typing.Self'):
        if isinstance(other, UTexture):
            other = other.texture

        self.texture[..., :3] *= other
        return self

    def __ior__(self, other: 'typing.Self'):
        """Union masks."""
        assert self.texture_type is bool
        assert other.texture_type is bool
        self.texture |= other.texture
        return self

    def __or__(self, other: 'typing.Self'):
        """Union masks."""
        assert self.texture_type is bool
        assert other.texture_type is bool

        texture = self.texture | other.texture
        return UTexture(self.width, self.height, texture, texture_type=bool)

    def __iand__(self, other: 'typing.Self'):
        """Inplace isect masks."""
        assert self.texture_type is bool
        assert other.texture_type is bool
        self.texture &= other.texture
        return self

    def __and__(self, other: 'typing.Self'):
        """Isect masks."""
        assert self.texture_type is bool
        assert other.texture_type is bool

        texture = self.texture & other.texture
        return UTexture(self.width, self.height, texture, texture_type=bool)

    def __setitem__(self, key, value):
        if isinstance(key, UTexture):
            self.texture[key.texture] = value
        else:
            self.texture[key] = value

    def fill_wrapped_rect(self, y0: int, y1: int, x0: int, x1: int, value: bool=True):
        assert self.texture_type is bool

        for y_slice in self.wrap_slices_h(y0, y1):
            for x_slice in self.wrap_slices_w(x0, x1):
                self.texture[y_slice, x_slice] = value

    def wrap_slices_w(self, start, stop):
        """ Returns wrapped slices fof width."""
        length = stop - start
        s0 = start % self.width
        s1 = s0 + length

        if s1 <= self.width:
            return [slice(s0, s1)]
        else:
            return [slice(s0, self.width), slice(0, s1 - self.width)]

    def wrap_slices_h(self, start, stop):
        """ Returns wrapped slices for height."""
        length = stop - start
        s0 = start % self.height
        s1 = s0 + length

        if s1 <= self.height:
            return [slice(s0, s1)]
        else:
            return [slice(s0, self.height), slice(0, s1 - self.height)]

    def draw_vline_wrapped(self, x, thickness, value=True):
        assert self.texture_type is bool
        half = thickness // 2

        x0 = x - half
        x1 = x + half + (thickness & 1)

        for xs in self.wrap_slices_w(x0, x1):
            self.texture[:, xs] = value

    def draw_hline_wrapped(self, y, thickness, value=True):
        assert self.texture_type is bool
        half = thickness // 2

        y0 = y - half
        y1 = y + half + (thickness & 1)

        for ys in self.wrap_slices_h(y0, y1):
            self.texture[ys, :] = value