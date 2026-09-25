# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

# Everything that happens below is thanks to the K-410
# The code was taken and modified from the 'btypes' module: https://github.com/K-410/btypes/tree/fafc510bd9de3aa3201edf5ad9bced26a5298bc0

import bpy
import typing
import ctypes
import platform
from ctypes import (
    POINTER,
    Union,
    Structure,
    c_float,
    c_short,
    c_int,
    c_int8,
    c_uint8,
    c_uint,
    c_long,
    c_int64,
    c_char,
    # cast,
    c_void_p,
    c_size_t,
    c_bool,
    sizeof,
    addressof
)

from . import bbox
from mathutils import Vector

version = bpy.app.version
bpy_struct_subclass = typing.TypeVar('bpy_struct_subclass', bound=bpy.types.bpy_struct)


class PyObject_HEAD(Structure):
    _fields_ = (("ob_refcnt", c_long),
                ("ob_type", c_void_p)
                )

class PyObject_VAR_HEAD(Structure):
    _fields_ = (("ob_refcnt", c_long),
                ("ob_type", c_void_p),
                ("ob_size", c_long)
                )

def info_(self):
    print('', '=' * 80, '\n', self)
    name_, size_ = 'Name', 'Size'
    print(f"{name_: <17}{size_: ^11}Offset   Value")

    typ = type(self)
    total_size = 0
    for name, *dtype in self._fields_:
        size = sizeof(*dtype)
        total_size += size

        try:
            safe_mem_read(addressof(self), total_size)
            value = getattr(self, name)
            if isinstance(value, ctypes.Array):
                value = tuple(value)
        except MemoryError:
            value = "!!!MemoryError!!!"

        print(f"{name[:20]: <20}{size: <6}{getattr(typ, name).offset: < 6}   {value}")

    print('\n', f'{total_size=}', '\n', '=' * 80)

def format_to_cpp_pointer(pointer) -> str:
    return hex(addressof(pointer)).upper()[2:].zfill(16)


class StructBase(Structure):
    _subclasses = []
    __annotations__ = {}

    def __init_subclass__(cls):
        def info_size(cls=cls):  # noqa
            info_(cls)
        setattr(cls, 'info', info_size)
        cls._subclasses.append(cls)

    def __new__(cls, srna: bpy_struct_subclass | None =None):
        if srna is None:
            return super().__new__(cls)
        try:
            return cls.from_address(srna.as_pointer())
        except AttributeError:
            raise Exception("Not a StructRNA instance")

    # Required
    def __init__(self, *_):  # noqa
        pass

    @staticmethod
    def _init_structs():
        """ Initialize subclasses, converting annotations to fields. """
        functype = type(lambda: None)

        for cls in StructBase._subclasses:
            fields = []
            anons = []
            for key, value in cls.__annotations__.items():
                if isinstance(value, functype):
                    value = value()
                elif isinstance(value, Union):
                    anons.append(key)
                fields.append((key, value))

            if anons:
                cls._anonynous_ = anons

            if fields:  # Base classes might not have _fields_. Don't set anything.
                cls._fields_ = fields
            cls.__annotations__.clear()

        StructBase._subclasses.clear()

    @classmethod
    def get_fields(cls, tar: bpy_struct_subclass):
        return cls.from_address(tar.as_pointer())

    @classmethod
    def get_fields_from_pyobj(cls, py_obj):
        return cls.from_address(id(py_obj))


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
            BArray = cls  # noqa

        else:
            class BArray(Structure):  # noqa
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
        self.from_address = self.inline_buffer_._type_.from_address  # noqa # pylint: disable=protected-access
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
        return self.inline_buffer_._type_.from_address((self.data_ + 4) + 16 * i)  # noqa # pylint: disable=protected-access

    def info(self):
        info_(self)


class BVector(StructBase):
    _fields_ = (("begin", c_void_p),
                ("end", c_void_p),
                ("capacity_end", c_void_p),
                ("_pad", c_char * 32))  # noqa

    _cache = {}

    def __new__(cls, c_type=None, inline_size=0):
        if inline_size == 0:
            inline_size = 4 if ctypes.sizeof(c_type) < 100 else 0
        if (c_type, inline_size) in cls._cache:
            return cls._cache[(c_type, inline_size)]

        elif c_type is None:
            assert inline_size == 1
            raise NotImplementedError
            BVector = cls  # noqa
        else:
            class BVector(Structure):  # noqa
                __name__ = __qualname__ = f"BVector{cls.__qualname__}"
                _fields_ = [("begin", c_void_p),
                            ("end", c_void_p),
                            ("capacity_end", POINTER(c_type)),

                            # ("allocator_", c_void_p),  # TODO: Check old versions and implement [[no_unique_address]]
                            ("inline_buffer_", c_char * (ctypes.sizeof(c_type) * inline_size))
                            ]

                if bpy.app.build_type != b"Release":
                    _fields_.append(("debug_size_", c_int64))
                __len__ = cls.__len__
                __iter__ = cls.__iter__
                __next__ = cls.__next__
                __getitem__ = cls.__getitem__
                __str__ = cls.__str__
                to_list = cls.to_list
        return cls._cache.setdefault((c_type, inline_size), BVector)

    def __len__(self):
        if self.begin and self.end:
            typ = self.capacity_end._type_  # noqa # pylint: disable=protected-access
            return (self.end - self.begin) // sizeof(typ)
        else:
            return 0

    def __iter__(self):
        self.value = 0
        typ = self.capacity_end._type_  # noqa # pylint: disable=protected-access
        self.from_address = POINTER(typ).from_address
        return self

    def __next__(self):
        if self.value >= len(self):
            raise StopIteration

        typ = self.capacity_end._type_  # noqa # pylint: disable=protected-access
        offset = self.value * sizeof(typ)
        address = self.begin + offset

        safe_mem_read(address, sizeof(typ))

        ret = ctypes.cast(address, POINTER(typ)).contents
        safe_mem_read(addressof(ret))
        self.value += 1
        return ret

    def __getitem__(self, i):
        if i < 0:
            i = len(self) + i
        if i < 0 or i >= len(self):
            raise IndexError(f'vector index {i} out of range')
        typ = self.capacity_end._type_  # noqa # pylint: disable=protected-access
        from_address = POINTER(typ).from_address
        return from_address(self.begin + sizeof(typ) * i).contents


    def __str__(self):
        typ = self.capacity_end._type_  # noqa # pylint: disable=protected-access
        try:
            safe_mem_read(self.begin)
            safe_mem_read(self.end)


            size = (self.end - self.begin) // ctypes.sizeof(typ)
        except MemoryError:
            size = -1

        return f"Vector[{typ.__qualname__}], {size = }"


    def to_list(self):
        from_address = POINTER(self.capacity_end._type_).from_address  # noqa # pylint: disable=protected-access
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
            ListBase = cls  # noqa

        else:
            class ListBase(Structure):  # noqa
                __name__ = __qualname__ = f"ListBase{cls.__qualname__}"
                _fields_ = (("first", POINTER(c_type)),
                            ("last",  POINTER(c_type)))
                __iter__ = cls.__iter__
                __bool__ = cls.__bool__
                __getitem__ = cls.__getitem__
                __str__ = cls.__str__
        return cls._cache.setdefault(c_type, ListBase)

    def __iter__(self):
        links_p = []
        # Some only have "last" member assigned, use it as a fallback.
        elem_n = self.first or self.last
        elem_p = elem_n and elem_n.contents.prev
        # elem_p = elem_n and safe_mem_read(addressof(elem_n)) and elem_n.contents.prev

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

    def __str__(self):
        first = self.first if self.first else "nullptr"
        last = self.last if self.last else "nullptr"
        return f"ListBase[{first}, {last}]"


class string(Structure):
    _fields_ = (("data", c_void_p),
                ("size",  c_size_t),
                ("capacity",  c_size_t),
                ("allocator", c_void_p),
                ("_additional_field",  c_size_t),  # TODO: This is valid or Vector has aligns ?
                )

    # _fields_ = (("data", c_char*32),)

    def __str__(self):
        BUFF_SIZE = 15
        if self.allocator > BUFF_SIZE:
            # TODO: Test large string and implement for gcc and clang, see: https://devblogs.microsoft.com/oldnewthing/20240510-00/?p=109742
            try:
                safe_mem_read(self.data, self.size)
                return ctypes.string_at(self.data, self.size).decode("utf-8", errors="replace")
            except MemoryError:
                return "!!!Memory Error!!!"

        ptr = addressof(self) + 8
        return ctypes.string_at(ptr, self.allocator).decode("utf-8", errors="replace")


#
class AncestorPointerRNA(Structure):
   _fields_ = (("type", c_void_p), ("data", c_void_p))

ANCESTOR_POINTER_RNA_DEFAULT_SIZE = 2
class PointerRNA(StructBase):
    owner_id: c_void_p
    type: c_void_p
    data: c_void_p
    # noinspection PyTypeHints
    ancestors: BVector(AncestorPointerRNA, ANCESTOR_POINTER_RNA_DEFAULT_SIZE)


def safe_mem_read(address, size=8):  # noqa
    return True

if platform.system() == "Windows":
    # safe_memory_read
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    ReadProcessMemory = kernel32.ReadProcessMemory
    ReadProcessMemory.argtypes = [
        ctypes.c_void_p,  # hProcess
        ctypes.c_void_p,  # lpBaseAddress
        ctypes.c_void_p,  # lpBuffer
        ctypes.c_size_t,  # nSize
        ctypes.POINTER(ctypes.c_size_t),  # lpNumberOfBytesRead
    ]
    ReadProcessMemory.restype = ctypes.c_bool

    GetCurrentProcess = kernel32.GetCurrentProcess
    GetCurrentProcess.restype = ctypes.c_void_p


    def safe_mem_read(address, size=8):
        buffer = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t()

        result = ReadProcessMemory(
            GetCurrentProcess(),
            ctypes.c_void_p(address),
            buffer,
            size,
            ctypes.byref(bytes_read),
        )

        if not result or bytes_read.value != size:
            raise MemoryError

        return buffer.raw
elif platform.system() == "Linux":
    def safe_mem_read(address, size=8):
        try:
            with open("/proc/self/mem", "rb", buffering=0) as f:
                f.seek(address)
                data = f.read(size)

            return data if len(data) == size else None
        except (OSError, ValueError):
            return None
elif platform.system() == "Darwin":
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

        mach_task_self = libc.mach_task_self
        mach_task_self.restype = ctypes.c_uint

        mach_vm_read_overwrite = libc.mach_vm_read_overwrite
        mach_vm_read_overwrite.argtypes = [
            ctypes.c_uint,  # target_task
            ctypes.c_uint64,  # address
            ctypes.c_uint64,  # size
            ctypes.c_uint64,  # data
            ctypes.POINTER(ctypes.c_uint64),  # outsize
        ]
        mach_vm_read_overwrite.restype = ctypes.c_int


        def safe_mem_read(address, size=8):
            buffer = ctypes.create_string_buffer(size)
            out_size = ctypes.c_uint64()

            result = mach_vm_read_overwrite(
                mach_task_self(),
                address,
                size,
                addressof(buffer),
                ctypes.byref(out_size),
            )

            if result != 0 or out_size.value != size:
                return None

            return buffer.raw
    except: # noqa
        import traceback
        traceback.print_exc()
        print(f"UniV: Can not create `safe_mem_read` function. Unsafe memory access in some types.")
else:
    print(f"UniV: Unknow platform { platform.system()!r}. Unsafe memory access in some types.")


class rctf(StructBase, bbox.BBox):
    xmin: c_float
    xmax: c_float
    ymin: c_float
    ymax: c_float


class rcti(StructBase, bbox.BBox):
    xmin: c_int
    xmax: c_int
    ymin: c_int
    ymax: c_int

    def __str__(self):
        return f"xmin={self.xmin}, xmax={self.xmax}, ymin={self.ymin}, ymax={self.ymax}, width={self.width}, height={self.height}"


class View2D(StructBase):
    tot: rctf
    cur: rctf
    vert: rcti
    hor: rcti
    mask: rcti

    min: c_float * 2  # noqa
    max: c_float * 2  # noqa

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

    if version >= (4, 0, 0):
        _pad6: c_char * 2  # noqa
        page_size_y: c_float
    else:
        _pad6: c_char * 6  # noqa

    sms: c_void_p  # SmoothView2DStore
    smooth_timer: c_void_p  # wmTimer

    @classmethod
    def get_rect(cls, view):
        return cls.from_address(view.as_pointer()).cur

    @classmethod
    def get_scale(cls, view):
        v2d = cls.from_address(view.as_pointer())
        return Vector((v2d.mask.width / v2d.cur.width, v2d.mask.height / v2d.cur.height))

    @classmethod
    def get_zoom(cls, view):
        v2d = cls.from_address(view.as_pointer())
        return (v2d.mask.xmax - v2d.mask.xmin) / (v2d.cur.xmax - v2d.cur.xmin)  # noqa

# source/blender/makesdna/DNA_ID.h | rev 362


class ID_Runtime_Remap(StructBase):
    status:                 c_int
    skipped_refcounted:     c_int
    skipped_direct:         c_int
    skipped_indirect:       c_int


# source/blender/makesdna/DNA_ID.h | rev 362
class ID_Runtime(StructBase):
    remap: ID_Runtime_Remap
    depsgraph:                  c_void_p
    _pad:                  c_void_p


class ID(StructBase):
    next:                   c_void_p
    prev:                   c_void_p
    # noinspection PyTypeHints
    newid: lambda: POINTER(ID)
    lib:                    c_void_p  # Library
    asset_data:             c_void_p  # AssetMetaData

    name:                   c_char * 66  # MAX_ID_NAME  # noqa
    flag:                   c_short
    tag:                    c_int
    us:                     c_int
    icon_id:                c_int
    recalc:                 c_uint
    recalc_up_to_undo_push: c_uint
    recalc_after_undo_push: c_uint

    session_uuid:           c_uint

    properties:             c_void_p  # IDProperty
    override_library:       c_void_p  # IDOverrideLibrary
    # noinspection PyTypeHints
    orig_id: lambda: POINTER(ID)
    py_instance:            c_void_p
    library_weak_reference: c_void_p
    runtime:                ID_Runtime


class ImageUser(StructBase):
    scene: c_void_p
    framenr: c_int
    frames: c_int
    offset: c_int
    sfra: c_int
    cycl: c_char
    multiview_eye: c_char
    pass_: c_short
    tile: c_int
    multi_index: c_short
    view: c_short
    layer: c_short
    flag: c_short


class Histogram(StructBase):
    pad: c_char * 5160  # noqa


class Scopes(StructBase):
    pad: c_char * 5272  # noqa


class SpaceImage(StructBase):
    next: c_void_p
    prev: c_void_p
    regionbase: ListBase
    spacetype: c_char
    link_flag: c_char
    _pad0: c_char * 6  # noqa
    image: c_void_p
    iuser: ImageUser
    scopes: Scopes
    sample_line_hist: Histogram
    gpd: c_void_p
    cursor: c_float * 2  # noqa
    xof: c_float
    yof: c_float
    zoom: c_float
    centx: c_float
    centy: c_float


class LayoutPanelHeader(StructBase):
    start_y: c_float
    end_y: c_float
    open_owner_ptr: PointerRNA
    open_prop_name: string


class LayoutPanelBody(StructBase):
    start_y: c_float
    end_y: c_float


# noinspection PyTypeHints
class LayoutPanels(StructBase):
    # _fields_ = (("headers", BVector(LayoutPanelHeader)), ("bodies", BVector(LayoutPanelBody)))
    headers: BVector(LayoutPanelHeader)
    bodies: BVector(LayoutPanelBody)


# noinspection PyTypeHints
class Panel_Runtime(StructBase):
    region_ofsx: c_int
    custom_data_ptr: POINTER(PointerRNA)
    block: c_void_p # uiBlock
    context: c_void_p  # bContextStore
    layout_panels: LayoutPanels

# noinspection PyTypeHints
class LayoutPanelState(StructBase):
    next: lambda: POINTER(LayoutPanelState)
    prev: lambda: POINTER(LayoutPanelState)
    # Identifier of the panel.
    idname: ctypes.c_char_p
    flag: c_uint8
    _pad: c_char * 3

    # A logical time set from #layout_panel_states_clock when the panel is used by the UI. This is
    # used to detect the least-recently-used panel states when some panel states should be removed.
    last_used: ctypes.c_uint32

# noinspection PyTypeHints
class Panel(StructBase):
    next: lambda: POINTER(Panel)
    prev: lambda: POINTER(Panel)
    
    # Runtime.
    type: c_void_p # PanelType
    # Runtime for drawing.
    layout: c_void_p #Layout
    panelname: c_char * 64
    # Panel name is identifier for restoring location.
    drawname: ctypes.c_char_p
    # Offset within the region.
    ofsx: c_int
    ofsy: c_int
    # Panel size including children. */
    sizex: c_int
    sizey: c_int
    # Panel size excluding children. */
    blocksizex: c_int
    blocksizey: c_int
    labelofs: c_short
    flag: c_short  # ePanel_Flag
    runtime_flag: c_short
    _pad: c_char * 6
    # Panels are aligned according to increasing sort-order. */
    sortorder: c_int
    # Runtime for panel manipulation.
    activedata: c_void_p
    # Sub panels.
    children: lambda: ListBase(Panel)


    #  This stores the open-close-state of layout-panels created with
    # `layout.panel(...)` in Python. For more information on layout-panels, see
    # `ui::Layout::panel_prop`.

    layout_panel_states: ListBase(LayoutPanelState)

    # This is increased whenever a layout panel state is used by the UI. This is used to allow for
    # some garbage collection of panel states when #layout_panel_states becomes large. It works by
    # removing all least-recently-used panel states up to a certain threshold.

    if version >= (4, 5, 0):
        layout_panel_states_clock: ctypes.c_uint32
        _pad2: c_char*4

    runtime: POINTER(Panel_Runtime)

# noinspection PyTypeHints
class PanelCategoryStack(StructBase):
    next: lambda: POINTER(PanelCategoryStack)
    prev: lambda: POINTER(PanelCategoryStack)
    idname: c_char * 64  # noqa


# noinspection PyTypeHints
class PanelCategoryDyn(StructBase):
    next: lambda: POINTER(PanelCategoryStack)
    prev: lambda: POINTER(PanelCategoryStack)
    idname: c_char * 64  # noqa
    rect: rcti


# noinspection PyTypeHints
# source/blender/makesdna/DNA_screen_types.h | rev 362
class ARegion(StructBase):
    next: lambda: POINTER(ARegion)
    prev: lambda: POINTER(ARegion)

    view2D: View2D
    winrct: rcti

    if version <= (4, 3, 2):
        drawrct: rcti

    winx: c_short
    winy: c_short

    if version > (3, 5, 0):
        category_scroll: c_int
        if version <= (4, 3, 2):
            _pad0: c_char * 4  # noqa

    if version <= (4, 3, 2):
        visible: c_short

    regiontype: c_short
    alignment: c_short
    flag: c_short

    sizex: c_short
    sizey: c_short

    if version <= (4, 3, 2):
        do_draw: c_short
        do_draw_overlay: c_short
    overlap: c_short
    flagfullscreen: c_short

    if version <= (4, 3, 2):
        type: c_void_p  # ARegionType
        uiblocks: ListBase
    else:
        _pad0: c_char * 2  # noqa

    panels: ListBase(Panel)
    panels_category_active: ListBase(PanelCategoryStack)
    ui_lists: ListBase
    ui_previews: ListBase

    if version <= (4, 3, 2):
        handlers: ListBase
        panels_category: ListBase(PanelCategoryDyn)
    else:
        view_states: ListBase

    if version <= (4, 3, 2):
        gizmo_map: c_void_p
        regiontimer: c_void_p
        draw_buffer: c_void_p

        headerstr: c_void_p
    if version >= (5, 1, 0):
        textbox_states: ListBase#(uiTextboxStateLink)
    regiondata: c_void_p

    # runtime: ARegion_Runtime

    @staticmethod
    def get_n_panel_from_area(_area: bpy.types.Area):
        for reg in _area.regions:
            if reg.type == 'UI':
                return reg
        raise AttributeError('Area not have N-Panel')

    @staticmethod
    def set_active_category(name: str, area: bpy.types.Area) -> bool:
        if bpy.app.version >= (4, 2, 0):
            reg = next(r for r in area.regions if r.type == 'UI')
            try:
                if reg.active_panel_category != name:
                    reg.active_panel_category = name
                return True
            except NameError:
                return False
        else:
            name = name.encode('utf-8')
            c_region = ARegion.get_fields(ARegion.get_n_panel_from_area(area))
            # Checking for a category in the N-Panel
            if not any(category for category in c_region.panels_category if category.idname == name):
                available_categories = [category.idname.decode("utf-8") for category in c_region.panels_category]
                if c_region.alignment == 1:
                    raise AttributeError('N-Panel aligned, cannot be set active category')
                raise AttributeError(f'Category \'{name.decode("utf-8")}\' not found in {available_categories}')

            # Check for the possibility to set an active category (for the presence of an allocated memory cell)
            category_history = [category for category in c_region.panels_category_active]
            if not category_history:
                raise AttributeError(
                    f'Unable to set a category because Blender did not allocate memory for active panels')

            # Check that the active panel with the given name is already active
            if category_history[0].idname == name:
                return False
            # If history length == 1, set it to
            if len(category_history) == 1:
                category_history[0].idname = name
                return True

            # Swap
            category_from_history = None
            for category in category_history:
                if category.idname == name:
                    category_from_history = category
                    break

            if not category_from_history:
                category_history[0].idname = name
                return True

            category_from_history.idname = category_history[0].idname
            category_history[0].idname = name
            return True


# source/blender/makesdna/DNA_windowmanager_types.h | rev 362
class ReportList(StructBase):
    list:           ListBase  # Report
    printlevel:     c_int
    storelevel:     c_int
    flag:           c_int
    _pad4:          c_char * 4  # noqa
    reporttimer:    c_void_p  # wmTimer

# noinspection PyTypeHints
class UndoStep(StructBase):
    next: lambda: POINTER(UndoStep)
    prev: lambda: POINTER(UndoStep)
    name: c_char * 64  # noqa
    type: c_void_p

    data_size:          c_size_t
    skip:               c_bool
    use_memfile_step:   c_bool
    use_old_bmain_data: c_bool
    is_applied:         c_bool

# noinspection PyTypeHints
class UndoStack(StructBase):
    steps: ListBase(ListBase)
    step_active: POINTER(UndoStep)
    step_active_memfile: POINTER(UndoStep)
    step_init: POINTER(UndoStep)
    group_level: c_int

# source/blender/makesdna/DNA_windowmanager_types.h | rev 362
class wmWindowManager(StructBase):
    # noinspection PyTypeHints
    ID: lambda: ID

    windrawable:                c_void_p
    winactive:                  c_void_p
    windows:                    ListBase

    initialized:                c_short
    file_saved:                 c_short
    op_undo_depth:              c_short

    outliner_sync_select_dirty: c_short

    operators:                  ListBase  # Operator undo history

    notifier_queue:             ListBase

    if version > (3, 2, 2):
        notifier_queue_set:     c_void_p  # GSet

    reports:                    ReportList
    jobs:                       ListBase
    paintcursors:               ListBase
    drags:                      ListBase
    keyconfigs:                 ListBase

    defaultconf:                c_void_p  # wmKeyConfig
    addonconf:                  c_void_p  # wmKeyConfig
    userconf:                   c_void_p  # wmKeyConfig

    timers:                     ListBase
    autosavetimer:              c_void_p  # wmTimer
    undo_stack:                 c_void_p  # UndoStack
    is_interface_locked:        c_char
    _pad7:                      c_char * 7  # noqa
    message_bus:                c_void_p  # wmMsgBus


# noinspection PyTypeHints
# source\blender\windowmanager\wm_event_system.h
class wmEventHandler(StructBase):
    next: lambda: POINTER(wmEventHandler)
    prev: lambda: POINTER(wmEventHandler)
    type: c_int
    flag: c_char
    poll: c_void_p

# noinspection PyTypeHints
# source\blender\makesdna\DNA_windowmanager_types.h
class wmWindow(StructBase):
    next: lambda: POINTER(wmWindow)
    prev: lambda: POINTER(wmWindow)

    ghostwin: c_void_p
    gpuctx: c_void_p

    parent: lambda: POINTER(wmWindow)

    scene: c_void_p
    new_scene: c_void_p
    view_layer_name: c_char * 64

    if version >= (3, 3):
        unpinned_scene: c_void_p  # Scene

    workspace_hook: c_void_p
    global_areas: ListBase * 3  # ScrAreaMap

    screen: c_void_p  # bScreen  # (deprecated)

    winid: c_int

    pos: c_short * 2
    size: c_short * 2
    windowstate: c_char
    active: c_char

    cursor: c_short
    lastcursor: c_short
    modalcursor: c_short
    grabcursor: c_short

    if version >= (3, 5, 0):
        pie_event_type_lock: c_short
        pie_event_type_last: c_short

    if version < (4, 5, 0):
        addmousemove: c_char
    tag_cursor_refresh: c_char

    event_queue_check_click: c_char
    event_queue_check_drag: c_char
    event_queue_check_drag_handled: c_char

    if version < (3, 5, 0):
        _pad0: c_char * 1
    else:
        if version < (4, 5, 0):
            event_queue_consecutive_gesture_type: c_char
        else:
            event_queue_consecutive_gesture_type: c_short
        event_queue_consecutive_gesture_xy: c_int * 2
        event_queue_consecutive_gesture_data: c_void_p  # wmEvent_ConsecutiveData

    if version < (3, 5, 0):
        pie_event_type_lock: c_short
        pie_event_type_last: c_short

    eventstate: c_void_p
    event_last_handled: c_void_p
    if version < (4, 5, 0):
        ime_data: c_void_p  # wmIMEData
    if version >= (4, 1, 0):
        if version < (4, 5, 0):
            ime_data_is_composing: c_char
        else:
            addmousemove: c_char
        _pad1: c_char * 7

    if version < (4, 5, 0):
        event_queue: ListBase
    handlers: ListBase(wmEventHandler)
    modalhandlers: ListBase(wmEventHandler)
    gesture: ListBase
    stereo3d_format: c_void_p
    drawcalls: ListBase
    cursor_keymap_status: c_void_p

    if version >= (4, 5, 0):
        _pad2: c_void_p

    if version >= (4, 1, 0):
        eventstate_prev_press_time_ms: c_size_t

    if version >= (4, 5, 0):
        runtime: c_void_p
        _pad3: c_void_p

class context(StructBase):  # Anonymous
    # noinspection PyTypeHints
    win: lambda: POINTER(wmWindow)
    area: c_void_p  # ScrArea ptr
    region: c_void_p  # ARegion ptr
    region_type: c_short

# noinspection PyTypeHints
class wmEvent(StructBase):
    next: lambda: POINTER(wmEvent)
    prev: lambda: POINTER(wmEvent)

    # Event code itself (short, is also in key-map).
    type: c_short
    # Press, release, scroll-value.
    val: c_short
    # Mouse pointer position, screen coord.
    xy: c_int * 2
    # Region relative mouse position (name convention before Blender 2.5).
    mval: c_int * 2
    # A single UTF8 encoded character.
    utf8_buf: c_char * 6
    # Modifier states: #KM_SHIFT, #KM_CTRL, #KM_ALT, #KM_OSKEY & #KM_HYPER.
    modifier: c_uint8
    # The direction (for #KM_PRESS_DRAG events only).
    direction: c_char
    # Raw-key modifier (allow using any key as a modifier).
    # Compatible with values in `type`.
    keymodifier: c_short
    # ...

# source\blender\windowmanager\wm_event_system.h
class wmEventHandler_Op(StructBase):
    head: wmEventHandler
    op: c_void_p  # wmOperator
    is_file_select: c_bool
    context: context

# noinspection PyTypeHints
class wmKeyMapItem(StructBase):
    next: lambda: POINTER(wmKeyMapItem)
    prev: lambda: POINTER(wmKeyMapItem)

    idname: c_char * 64
    properties: c_void_p
    propvalue_str: c_char * 64
    propvalue: c_short
    type: c_short
    val: c_int8
    direction: c_int8

    if bpy.app.version < (4, 5, 0):
        shift: c_short
        ctrl: c_short
        alt: c_short
        oskey: c_short
    else:
        shift: c_int8
        ctrl: c_int8
        alt: c_int8
        oskey: c_int8
        hyper: c_int8

        _pad0: c_char * 7

    keymodifier: c_short

    if bpy.app.version < (4, 5, 0):
        flag: c_short
        maptype: c_short
    else:
        flag: c_uint8
        maptype: c_uint8

    # Unique identifier. Positive for kmi that override builtins, negative otherwise.
    id: c_short
    if bpy.app.version < (4, 5, 0):
        _pad0: c_char * 2
    ptr: c_void_p


# noinspection PyTypeHints
class CustomDataLayer(StructBase):
    type: c_int
    offset: c_int
    flag: c_int
    active: c_int
    active_rnd: c_int

    if version < (5, 1, 0):
        active_clone: c_int
        active_mask: c_int

    uid: c_int
    if version >= (3, 5, 0):
        name: c_char * 68
        _pad1: c_char * 4
    else:
        name: c_char * 64
    data: c_void_p

    sharing_info: c_void_p


# noinspection PyTypeHints
class CustomData(StructBase):
    layers: lambda: POINTER(CustomDataLayer)

    if version >= (3, 4, 0):
        typemap: c_int * 53
    else:
        if version >= (3, 2, 0):
            typemap: c_int * 52
        else:
            typemap: c_int * 50
        _pad1: c_char * 4

    totlayer: c_int
    maxlayer: c_int

    totsize: c_int

    pool: c_void_p
    external: c_void_p

    def get_offset(self, typ):
        return self.layers[self.get_active_layer_index(typ)].offset

    def get_active_layer_index(self, typ):
        layer_index = self.typemap[typ]
        assert layer_index != -1
        return layer_index + self.layers[layer_index].active


class CBMesh(StructBase):
    totvert: c_int
    totedge: c_int
    totloop: c_int
    totface: c_int
    totvertsel: c_int
    totedgesel: c_int
    totfacesel: c_int

    elem_index_dirty: c_char

    vpool: c_void_p
    epool: c_void_p
    lpool: c_void_p
    fpool: c_void_p

    vtable: c_void_p
    etable: c_void_p
    ftable: c_void_p

    vtable_tot: c_int
    etable_tot: c_int
    ftable_tot: c_int

    vtoolflagpool: c_void_p
    etoolflagpool: c_void_p
    ftoolflagpool: c_void_p

    use_toolflags: c_bool

    if version >= (5, 0, 0):
        uv_select_sync_valid: c_bool

    toolflag_index: c_int

    vdata: CustomData
    edata: CustomData
    ldata: CustomData
    pdata: CustomData


class BPy_BMLayerItem(StructBase):
    if version >= (5, 2, 0):
        pyhead: PyObject_HEAD
    else:
        pyhead: PyObject_VAR_HEAD
    # noinspection PyTypeHints
    bm: POINTER(CBMesh)
    htype: c_char
    type: c_int  # customdata type - CD_XXX
    index: c_int  # index of this layer type

class PyBMesh(StructBase):
    # Cleanup: use PyObject_HEAD for fixed-size types: https://projects.blender.org/blender/blender/pulls/155741
    if version >= (5, 2, 0):
        pyhead: PyObject_HEAD
    else:
        pyhead: PyObject_VAR_HEAD
    # noinspection PyTypeHints
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
        if bm.totfacesel:
            return bm.totfacesel == bm.totface
        return False

    @classmethod
    def is_full_face_deselected(cls, bm):
        return cls.fields(bm).totfacesel == 0

    @classmethod
    def is_full_edge_selected(cls, bm):
        bm = cls.fields(bm)
        if bm.totedgesel:
            return bm.totedgesel == bm.totedge
        return False

    @classmethod
    def is_full_edge_deselected(cls, bm):
        return cls.fields(bm).totedgesel == 0

    @classmethod
    def is_full_vert_selected(cls, bm):
        bm = cls.fields(bm)
        if bm.totvertsel:
            return bm.totvertsel == bm.totvert
        return False

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

class ImBufByteBuffer(StructBase):
    # noinspection PyTypeHints
    data: POINTER(c_uint8)  # uint8_t
    ownership: c_int  # ImBufOwnership
    colorspace: c_void_p  # ColorSpace

# noinspection PyTypeHints
class ImBuf(StructBase):
    # Width and Height of our image buffer.
    x: c_int
    y: c_int

    if version >= (5, 1, 0):
        display_size: c_int * 2
        data_offset: c_int * 2
        display_offset: c_int * 2

    #  Active amount of bits/bit-planes.
    planes: c_char

    #  Number of channels in `rect_float` (0 = 4 channel default)
    channels: c_int

    #   Controls which components should exist. */
    flags: c_int

    #  Image pixel buffer (8bit representation):
    #  - color space defaults to `sRGB`.
    #  - alpha defaults to 'straight'.

    byte_buffer: ImBufByteBuffer

class Py_ImBuf(StructBase):
    if version >= (5, 2, 0):
        pyhead: PyObject_HEAD
    else:
        pyhead: PyObject_VAR_HEAD
    # noinspection PyTypeHints
    ibuf: POINTER(ImBuf)

    @classmethod
    def get_fields(cls, py_ibuf) -> ImBuf:
        ibuf = cls.from_address(id(py_ibuf))
        # print(ctypes.c_void_p.from_address(c_bm.bm.contents))  # get address by int
        # ctypes.addressof(c_bm.bm.contents)
        return ibuf.ibuf.contents

StructBase._init_structs()  # noqa
