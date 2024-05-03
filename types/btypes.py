import sys
import bpy
import bmesh
import typing
from ctypes import (
    POINTER,
    Structure,
    c_float,
    c_short,
    c_int,
    c_long,
    c_int64,
    c_char,
    # cast,
    c_void_p,
    sizeof,
    # addressof
)
# bpy.ops.consoleclear.clear()
version = bpy.app.version


def info_(self):
    print('', '=' * 80, '\n', self)
    Name, Size = 'Name', 'Size'
    print(f"{Name: <17}{Size: ^11}Offset   Value")
    ofs = 0
    total_size = 0
    for name, *dtype in self._fields_:
        value = getattr(self, name)
        size = sizeof(*dtype)
        total_size += size
        print(f"{name[:20] : <20}{size : <6}{ofs : < 6}   {value}")
        ofs += size

    print('\n', f'{total_size=}', '\n', '=' * 80)

class StructBase(Structure):
    _subclasses = []
    __annotations__ = {}

    def __init_subclass__(cls):
        def info_size(cls=cls):  # noqa
            info_(cls)
        setattr(cls, 'info', info_size)
        cls._subclasses.append(cls)

    @staticmethod
    def _init_structs():
        """ Initialize subclasses, converting annotations to fields. """
        functype = type(lambda: None)

        for cls in StructBase._subclasses:
            fields = []
            for field, value in cls.__annotations__.items():
                if isinstance(value, functype):
                    value = value()
                fields.append((field, value))

            if fields:  # Base classes might not have _fields_. Don't set anything.
                cls._fields_ = fields
            cls.__annotations__.clear()

        StructBase._subclasses.clear()


class BArray(StructBase):
    _fields_ = (
        ("data_", c_void_p),
        ("size_", c_int64),
        ("allocator_", c_void_p),
        ("inline_buffer_", c_void_p),
    )

    _cache = {}

    def __new__(cls, c_type=None):
        if c_type in cls._cache:
            return cls._cache[c_type]

        elif c_type is None:
            BArray = cls

        else:
            class BArray(Structure):
                __name__ = __qualname__ = f"BArray{cls.__qualname__}"
                _fields_ = (
                    ("data_", c_void_p),
                    ("size_", c_int64),
                    ("allocator_", c_void_p),
                    ("inline_buffer_", POINTER(c_type)),
                )
                __len__ = cls.__len__
                __iter__ = cls.__iter__
                __next__ = cls.__next__
                __getitem__ = cls.__getitem__
                info = cls.info
                to_list = cls.to_list
        return cls._cache.setdefault(c_type, BArray)

    def __len__(self):
        return self.size_

    def __iter__(self):
        self.value = 4
        self.from_address = self.inline_buffer_._type_.from_address
        return self

    def __next__(self):
        if self.value < self.size_ * 16:
            ret = self.from_address(self.data_ + self.value)
            self.value += 16
            return ret
        else:
            raise StopIteration

    def __getitem__(self, i):
        if i < 0:
            i = self.size_ + i
        if i < 0 or i >= self.size_:
            raise IndexError(f'array index {i - self.size_ if i < 0 else i} out of range')
        return self.inline_buffer_._type_.from_address((self.data_ + 4) + 16 * i)

    def info(self):
        info_(self)

class BVector(StructBase):
    _fields_ = (("begin", c_void_p),
                ("end", c_void_p),
                ("capacity_end", c_void_p),
                ("_pad", c_char * 32))

    _cache = {}

    def __new__(cls, c_type=None):
        if c_type in cls._cache:
            return cls._cache[c_type]

        elif c_type is None:
            BVector = cls
        else:
            class BVector(Structure):
                __name__ = __qualname__ = f"BVector{cls.__qualname__}"
                _fields_ = (("begin", c_void_p),
                            ("end", c_void_p),
                            ("capacity_end", POINTER(c_type)),
                            ("_pad", c_char * 32))
                __len__ = cls.__len__
                __iter__ = cls.__iter__
                __next__ = cls.__next__
                __getitem__ = cls.__getitem__
                to_list = cls.to_list
        return cls._cache.setdefault(c_type, BVector)

    def __len__(self):
        return (self.end - self.begin) // 8

    def __iter__(self):
        self.value = 0
        self.from_address = POINTER(self.capacity_end._type_).from_address
        return self

    def __next__(self):
        if self.value < len(self):
            ret = self.from_address(self.begin + (self.value * 8)).contents
            self.value += 1
            return ret
        else:
            raise StopIteration

    def __getitem__(self, i):
        if i < 0:
            i = len(self) + i
        if i < 0 or i >= len(self):
            raise IndexError(f'vector index {i} out of range')
        from_address = POINTER(self.capacity_end._type_).from_address
        return from_address(self.begin + 8 * i).contents

    def to_list(self):
        from_address = POINTER(self.capacity_end._type_).from_address
        return [from_address(self.begin + (i * 8)).contents for i in range(len(self))]

class ListBase(Structure):
    """Generic linked list used throughout Blender.

    A typed ListBase class is defined using the syntax:
        ListBase(c_type)
    """
    _fields_ = (("first", c_void_p),
                ("last",  c_void_p))
    _cache = {}

    def __new__(cls, c_type=None):
        if c_type in cls._cache:
            return cls._cache[c_type]

        elif c_type is None:
            ListBase = cls

        else:
            class ListBase(Structure):
                __name__ = __qualname__ = f"ListBase{cls.__qualname__}"
                _fields_ = (("first", POINTER(c_type)),
                            ("last",  POINTER(c_type)))
                __iter__ = cls.__iter__
                __bool__ = cls.__bool__
                __getitem__ = cls.__getitem__
        return cls._cache.setdefault(c_type, ListBase)

    def __iter__(self):
        links_p = []
        # Some only have "last" member assigned, use it as a fallback.
        elem_n = self.first or self.last
        elem_p = elem_n and elem_n.contents.prev

        # Temporarily store reversed links and yield them in the right order.
        if elem_p:
            while elem_p:
                links_p.append(elem_p.contents)
                elem_p = elem_p.contents.prev
            yield from reversed(links_p)

        while elem_n:
            yield elem_n.contents
            elem_n = elem_n.contents.next

    def __getitem__(self, i):
        return list(self)[i]

    def __bool__(self):
        return bool(self.first or self.last)


class rectBase(StructBase):
    """Base for rcti, rctf"""

    @property
    def size_x(self) -> float:
        return self.xmax - self.xmin

    @property
    def size_y(self) -> float:
        return self.ymax - self.ymin

    @property
    def cent_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def cent_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    def get_position(self):
        return self.xmin, self.ymin

    def set_position(self, x, y):
        self.xmax -= self.xmin - x
        self.ymax -= self.ymin - y
        self.xmin = x
        self.ymin = y

    def translate(self, xy: typing.Sequence[float]):
        self.xmin += xy[0]
        self.ymin += xy[1]
        self.xmax += xy[0]
        self.ymax += xy[1]

    def scale(self, scale: typing.Sequence[float]):
        cent_x = self.cent_x
        cent_y = self.cent_y
        size_x_half = self.size_x * (scale[0] * 0.5)
        size_y_half = self.size_y * (scale[1] * 0.5)
        self.xmin = cent_x - size_x_half
        self.ymin = cent_y - size_y_half
        self.xmax = cent_x + size_x_half
        self.ymax = cent_y + size_y_half

    def __str__(self):
        return f"{self.xmax=} {self.xmin=} {self.ymax=} {self.ymin=}"

    def __contains__(self, pt):
        x, y = pt
        return self.xmin <= x <= self.xmax and \
               self.ymin <= y <= self.ymax  # noqa


class rctf(rectBase):
    xmin: c_float
    xmax: c_float
    ymin: c_float
    ymax: c_float


class rcti(rectBase):
    xmin: c_int
    xmax: c_int
    ymin: c_int
    ymax: c_int


class View2D(StructBase):
    tot: rctf
    cur: rctf
    vert: rcti
    hor: rcti
    mask: rcti

    min: c_float * 2
    max: c_float * 2

    minzoom: c_float
    maxzoom: c_float

    scroll: c_short
    scroll_ui: c_short

    keeptot: c_short
    keepzoom: c_short
    keepofs: c_short

    flag: c_short
    align: c_short

    winx: c_short
    winy: c_short
    oldwinx: c_short
    oldwiny: c_short

    around: c_short

    alpha_vert: c_char
    alpha_hor: c_char

    if version > (2, 92):
        _pad6: c_char * 6

    sms: c_void_p  # SmoothView2DStore
    smooth_timer: c_void_p  # wmTimer

    @classmethod
    def get_rect(cls, view):
        return cls.from_address(view.as_pointer()).cur


class CBMesh(StructBase):
    totvert: c_int
    totedge: c_int
    totloop: c_int
    totface: c_int
    totvertsel: c_int
    totedgesel: c_int
    totfacesel: c_int

class PyBMesh(StructBase):
    ob_refcnt: c_long
    ob_type: c_void_p
    ob_size: c_long
    bm: POINTER(CBMesh)

    @classmethod
    def fields(cls, bm):
        c_bm = cls.from_address(id(bm))
        # print(ctypes.c_void_p.from_address(c_bm.bm.contents))  # get address by int
        # ctypes.addressof(c_bm.bm.contents)
        return c_bm.bm.contents

    @classmethod
    def is_full_face_selected(cls, bm):
        bm = cls.fields(bm)
        return bm.totfacesel == bm.totface

    @classmethod
    def is_full_face_deselected(cls, bm):
        return cls.fields(bm).totfacesel == 0

    @classmethod
    def is_full_edge_selected(cls, bm):
        bm = cls.fields(bm)
        return bm.totedgesel == bm.totedge

    @classmethod
    def is_full_edge_deselected(cls, bm):
        return cls.fields(bm).totedgesel == 0

    @classmethod
    def is_full_vert_selected(cls, bm):
        bm = cls.fields(bm)
        return bm.totvertsel == bm.totvert

    @classmethod
    def is_full_vert_deselected(cls, bm):
        return cls.fields(bm).totvertsel == 0

    @classmethod
    def total_face_sel(cls, bm):
        return cls.fields(bm).totfacesel

    @classmethod
    def total_edge_sel(cls, bm):
        return cls.fields(bm).totedgesel

    @classmethod
    def total_vert_sel(cls, bm):
        return cls.fields(bm).totvertsel

    @classmethod
    def total_loop(cls, bm):
        return cls.fields(bm).totloop


StructBase._init_structs()  # noqa
