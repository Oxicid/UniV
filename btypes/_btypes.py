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
    c_double,
    c_short,
    c_int,
    c_int8,
    c_uint8,
    c_uint,
    c_long,
    c_int64,
    c_byte,
    c_ubyte,
    c_char,
    c_char_p,
    # cast,
    c_void_p,
    c_size_t,
    c_bool,
    sizeof,
    addressof,
    CFUNCTYPE
)

# from .. import utypes
from mathutils import Vector

version = bpy.app.version

is_debug_build = False
if bpy.app.build_type != b"Release":
    # Blender built with release flag, `bpy.app.build_type` might be empty. # TODO: Bugreport ?
    if bpy.app.build_type == b"Debug" or "Debug" in typing.__file__:
        is_debug_build = True

bpy_struct_subclass = typing.TypeVar('bpy_struct_subclass', bound=bpy.types.bpy_struct)

def factory(func):
    return func()


@factory
def event_type_to_string():
    return {
    e.value: e.identifier for e in bpy.types.Event.bl_rna.properties["type"].enum_items
}.__getitem__


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
                try:
                    cls._fields_ = fields
                except Exception as e:
                    print(f"Cant register {cls.__qualname__!r}.")
                    raise e
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

    def __new__(cls, c_type, inline_size=0):
        assert ctypes.sizeof(c_type) != 0, (f"Not found `_fields_`, maybe c_type {c_type.__qualname__!r} not initialized."
                                            f" Could lazy loading using a lambda be used?")
        if inline_size == 0:
            inline_size = 4 if ctypes.sizeof(c_type) < 100 else 0

        if (c_type, inline_size) in cls._cache:
            return cls._cache[(c_type, inline_size)]
        else:
            class BVector(Structure):  # noqa
                __name__ = __qualname__ = f"BVector{cls.__qualname__}"
                _fields_ = [("begin", c_void_p),
                            ("end", c_void_p),
                            ("capacity_end", POINTER(c_type)),

                            # ("allocator_", c_void_p),  # TODO: Check old versions and implement [[no_unique_address]]
                            ("inline_buffer_", c_char * (ctypes.sizeof(c_type) * inline_size))
                            ]

                if True: #is_debug_build:
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
             i += len(self)
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


class _Bxty(Union):
    _fields_ = (("buf", c_char * 16), ("ptr", c_void_p) )

# TODO: Test large string and implement for gcc and clang, see: https://devblogs.microsoft.com/oldnewthing/20240510-00/?p=109742
#  https://github.com/elliotgoodrich/SSO-23
class string(Structure):
    if platform.system() == "Windows":
        _fields_ = [
            ("bx", _Bxty),
            ("size", c_size_t),
            ("capacity", c_size_t)]
    else:
        _fields_ = [
            ("ptr", c_void_p),
            ("size", c_size_t),
            ("buf", c_char*16)
        ]
    if is_debug_build and platform.system() == "Windows":
        _fields_.append(("allocator", c_void_p))  # noqa

    if platform.system() == "Windows":
        @property
        def data_ptr(self):
            if self.capacity > 15:
                return self.bx.ptr
            return addressof(self.bx)

        def __str__(self):
            if self.size == 0 or self.size > 1000:
                return ""

            ptr = self.data_ptr
            try:
                return safe_mem_read(ptr, self.size).decode("utf-8", errors="replace")
            except MemoryError:
                return "!!!MemoryError!!!"
    else:
        def __str__(self):
            if self.size == 0 or self.size > 1000:
                return ""

            try:
                # TODO: Check on long strings
                return safe_mem_read(self.ptr, self.size).decode("utf-8", errors="replace")
            except MemoryError:
                return "!!!MemoryError!!!"

#
class AncestorPointerRNA(Structure):
   _fields_ = (("type", c_void_p), ("data", c_void_p))

   def info(self):
       info_(self)

ANCESTOR_POINTER_RNA_DEFAULT_SIZE = 2
class PointerRNA(StructBase):
    owner_id: c_void_p
    type: c_void_p
    data: c_void_p
    if platform.system() == "Windows":
        if version >= (5, 1, 0) or version <= (4, 4, 3):  # TODO: Check in other platforms
            data: c_void_p
    # noinspection PyTypeHints
    ancestors: BVector(AncestorPointerRNA, ANCESTOR_POINTER_RNA_DEFAULT_SIZE)

    def info(self):
        info_(self)

# # noinspection PyTypeHints
# # source/blender/makesrna/RNA_types.h | rev 362
# class PointerRNA(StructBase):
#     owner_id: l
#     type: c_void_p  # StructRNA
#     data: c_void_p


def safe_mem_read(address, size=8):  # noqa
    return " "

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

            if len(data) != size:
                raise MemoryError
            return data
        except (OSError, ValueError):
            raise MemoryError
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
                raise MemoryError

            return buffer.raw
    except: # noqa
        import traceback
        traceback.print_exc()
        print(f"UniV: Can not create `safe_mem_read` function. Unsafe memory access in some types.")
else:
    print(f"UniV: Unknow platform { platform.system()!r}. Unsafe memory access in some types.")


class vec2Base(StructBase):
    """Base for vec2i, vec2s, vec2f."""

    def __setitem__(self, i, val):
        setattr(self, ("x", "y")[i], val)

    # Allow subscript, but avoid for performance reasons
    def __getitem__(self, i):
        return getattr(self, ("x", "y")[i])

    def __iter__(self):
        return iter((self.x, self.y))


class vec2i(vec2Base):
    x: c_int
    y: c_int


class vec2s(vec2Base):
    x: c_short
    y: c_short


class vec2f(vec2Base):
    x: c_float
    y: c_float


from .. import utypes
class rctf(StructBase, utypes.bbox.BBox):
    xmin: c_float
    xmax: c_float
    ymin: c_float
    ymax: c_float


class rcti(StructBase, utypes.bbox.BBox):
    xmin: c_int
    xmax: c_int
    ymin: c_int
    ymax: c_int

    def __str__(self):
        return f"xmin={self.xmin}, xmax={self.xmax}, ymin={self.ymin}, ymax={self.ymax}, width={self.width}, height={self.height}"


# source/blender/makesdna/DNA_ID.h | rev 362
class ID_Runtime_Remap(StructBase):
    status:                 c_int
    skipped_refcounted:     c_int
    skipped_direct:         c_int
    skipped_indirect:       c_int


# source/blender/makesdna/DNA_ID.h
class ID_Runtime(StructBase):
    remap: ID_Runtime_Remap
    if version >= (4, 2, 0):
        depsgraph:          c_void_p
        _pad:               c_void_p



MAX_ID_NAME = 66
if version >= (5, 0, 0):
    MAX_ID_NAME = 258
# noinspection PyTypeHints
# source/blender/makesdna/DNA_ID.h
class ID(StructBase):
    next:                   c_void_p
    prev:                   c_void_p
    newid:                  lambda: POINTER(ID)
    lib:                    c_void_p  # Library
    if version >= (2, 92, 0):
        asset_data:         c_void_p  # AssetMetaData

    name:                   c_char * MAX_ID_NAME
    flag:                   c_short
    if version >= (2, 80, 0):
        tag:                c_int
    else:
        tag:                c_short
        pad_s1:             c_short
    us:                     c_int
    icon_id:                c_int
    if version >= (2, 80, 0):
        recalc:             c_uint

    if version < (2, 83, 0):
        if version == (2, 79, 0):
            properties: c_void_p
        else:
            _pad:               c_char * 4
            properties:         c_void_p
            override_library:   c_void_p

    if version >= (2, 83, 0):
        recalc_up_to_undo_push: c_uint
        recalc_after_undo_push: c_uint
        session_uuid:           c_uint

        if version >= (5, 0, 0):
            id_hash:            c_char * 16

        properties:             c_void_p  # IDProperty
        if version >= (4, 5, 0):
            system_properties:  c_void_p
            _pad1:  c_void_p

        override_library:       c_void_p  # IDOverrideLibrary

    if version >= (2, 80, 0):
        orig_id: lambda: POINTER(ID)
        py_instance:            c_void_p
    if version >= (2, 92, 0):
        library_weak_reference: c_void_p
    if version >= (3, 2, 0):
        if version >= (5, 0, 0):
            runtime:        c_void_p  # ID_RuntimeHandle
        else:
            runtime:        ID_Runtime

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
    headers: lambda : BVector(LayoutPanelHeader)
    bodies: lambda : BVector(LayoutPanelBody)

# # noinspection PyTypeHints
# # source/blender/makesdna/DNA_screen_types.h | rev 362
# class Panel_Runtime(StructBase):
#     region_ofsx: c_int
#     _pad4: c_char * 4
#
#     if version > (2, 83):
#         custom_data_ptr: lambda: POINTER(PointerRNA)
#         block: lambda: POINTER(uiBlock)
#
#     if version > (3, 1):
#         context: c_void_p  # bContextStore

# noinspection PyTypeHints
class Panel_Runtime(StructBase):
    region_ofsx: c_int
    custom_data_ptr: POINTER(PointerRNA)
    block: c_void_p # uiBlock
    context: c_void_p  # bContextStore
    layout_panels: LayoutPanels
    if version >= (5, 1, 0):
        layout_panel_states_storage: lambda: POINTER(ListBase(LayoutPanelState))


# source/blender/editors/include/UI_interface.h | rev 362
class uiBlockInteraction_CallbackData(StructBase):
    begin_fn: c_void_p  # uiBlockInteractionBeginFn
    end_fn: c_void_p  # uiBlockInteractionEndFn
    update_fn: c_void_p  # uiBlockInteractionUpdateFn
    arg1: c_void_p

# noinspection PyTypeHints
# source/blender/editors/interface/interface_intern.hh | rev 362
class uiPopupBlockCreate(StructBase):
    create_func: c_void_p  # uiBlockCreateFunc
    handle_create_func: c_void_p  # uiBlockHandleCreateFunc
    arg: c_void_p
    arg_free: c_void_p
    event_xy: vec2i
    butregion: lambda: POINTER(ARegion)
    but: lambda: POINTER(uiBut)


# source/blender/editors/interface/interface_intern.hh | rev 362
class uiKeyNavLock(StructBase):
    is_keynav: c_bool
    event_xy: vec2i


# source/blender/blenlib/BLI_vector.hh | rev 362
class blenderVector(StructBase):
    begin_: c_void_p
    end_: c_void_p
    capacity_end_: c_void_p

# noinspection PyTypeHints
# source/blender/editors/interface/interface_intern.hh | rev 362
class uiBlock(StructBase):
    next: lambda: POINTER(uiBlock)
    prev: lambda: POINTER(uiBlock)

    buttons: lambda: ListBase(uiBut)
    # ... (cont)

# noinspection PyTypeHints
# source/blender/editors/interface/interface_intern.hh | rev 362
class uiBut(StructBase):
    next: lambda: POINTER(uiBut)
    prev: lambda: POINTER(uiBut)

    if version > (2, 90):
        layout: c_void_p  # uiLayout

    flag: c_int
    drawflag: c_int
    type: c_int  # eButType
    pointype: c_int  # eButPointerType

    bit: c_short
    bitnr: c_short
    retval: c_short
    strwidth: c_short
    alignnr: c_short

    ofs: c_short
    pos: c_short
    selsta: c_short
    selend: c_short

    str: c_char_p
    strdata: c_char * 128  # UI_MAX_NAME_STR
    drawstr: c_char * 400  # UI_MAX_DRAW_STR

    rect: rctf
    poin: c_char_p

    hardmin: c_float
    hardmax: c_float
    softmin: c_float
    softmax: c_float

    a1: c_float
    a2: c_float
    col: c_ubyte * 4

    if version > (3, 1):
        identity_cmp_func: c_void_p

    func: c_void_p
    func_arg1: c_void_p
    func_arg2: c_void_p

    funcN: c_void_p

    if version > (2, 82):
        func_argN: c_void_p

    context: c_void_p

    autocomplete_func: c_void_p
    autofunc_arg: c_void_p

    if version < (2, 83):
        search_create_func: c_void_p
        search_func: c_void_p
        free_search_arg: c_bool
        search_arg: c_void_p

    rename_func: c_void_p
    rename_arg1: c_void_p
    rename_orig: c_void_p
    hold_func: c_void_p
    hold_argN: c_void_p

    tip: c_char_p
    tip_func: c_void_p
    tip_arg: c_void_p

    if version > (2, 93):
        tip_arg_free: c_void_p

    disabled_info: c_char_p

    icon: c_int  # BIFIconID

    if version < (2, 93):
        emboss: c_char
    else:
        emboss: c_int  # eUIEmbossType

    if version < (3, 2):
        pie_dir: c_byte
    else:
        pie_dir: c_int  # RadialDirection

    changed: c_bool
    unit_type: c_ubyte

    if version < (3, 3):
        modifier_key: c_short

    iconadd: c_short

    block_create_func: c_void_p
    menu_create_func: c_void_p
    menu_step_func: c_void_p

    rnapoin: lambda: PointerRNA
    # rnapoin: PointerRNA
    rnaprop: c_void_p  # PropertyRNA
    rnaindex: c_int

    if version < (2, 93):
        rnaserachpoin: c_void_p * 3
        rnasearchprop: c_void_p

    optype: lambda: POINTER(wmOperatorType)
    opptr: lambda: POINTER(PointerRNA)
    opcontext: c_int  # enum wmOperatorCallContext
    menu_key: c_ubyte
    extra_op_icons: ListBase  # uiButExtraOpIcon
    dragtype: c_char
    dragflag: c_short
    dragpoin: c_void_p
    imb: c_void_p  # ImBuf
    imb_scale: c_float
    active: c_void_p  # uiHandleButtonData
    custom_data: c_void_p
    editstr: c_char_p
    editval: POINTER(c_double)
    editvec: POINTER(c_float)

    if version < (2, 93):
        editcoba: c_void_p
        editcumap: c_void_p
        editprofile: c_void_p

    pushed_state_func: c_void_p  # uiButPushedStateFunc
    pushed_state_arg: c_void_p

    if version > (3, 3):
        # noinspection PyTypeHints
        class IconTextOverlay(StructBase):
            text: c_char * 5

        icon_overlay_text: IconTextOverlay
        _pad0: c_char * 3

    block: lambda: POINTER(uiBlock)


# noinspection PyTypeHints
# source/blender/editors/space_text/text_draw.c | rev 362
class DrawCache(StructBase):
    line_height: POINTER(c_int)
    total_lines: c_int
    nlines: c_int

    winx: c_int
    wordwrap: c_int
    showlnum: c_int
    tabnumber: c_int

    lheight: c_short
    cwidth_px: c_char
    text_id: c_char * 66  # MAX_ID_NAME

    update_flag: c_short
    valid_head: c_int
    valid_tail: c_int



# noinspection PyTypeHints
# source/blender/makesdna/DNA_space_types.h | rev 362
class SpaceText_Runtime(StructBase):
    # Confusingly not line height in pixels. Use property instead.
    _lheight_px: c_int

    cwidth_px: c_int
    scroll_region_handle: rcti
    scroll_region_select: rcti
    line_number_display_digits: c_int
    viewlines: c_int
    scroll_px_per_line: c_float
    scroll_ofs_px: vec2i
    _pad1: c_char * 4
    drawcache: lambda: POINTER(DrawCache)

    @property
    def lpad_px(self):
        return self.cwidth_px * (self.line_number_display_digits + 3)  # noqa

    @property
    def lheight_px(self):
        return int(self._lheight_px * 1.3)  # noqa

# noinspection PyTypeHints
# source/blender/makesdna/DNA_text_types.h | rev 362
class TextLine(StructBase):
    next: lambda: POINTER(TextLine)
    prev: lambda: POINTER(TextLine)

    line: c_char_p
    format: c_char_p
    len: c_int
    _pad0: c_char * 4

# noinspection PyTypeHints
# source/blender/makesdna/DNA_text_types.h | rev 362
class Text(StructBase):
    id: lambda: ID
    filepath: c_char_p
    compiled: c_void_p
    flags: c_int

    if version < (2, 90):
        nlines: c_int
    else:
        _pad0: c_char * 4

    lines: ListBase(TextLine)
    curl: POINTER(TextLine)
    sell: POINTER(TextLine)
    curc: c_int
    selc: c_int
    mtime: c_double

# noinspection PyTypeHints
# source/blender/editors/interface/interface_region_menu_popup.cc | rev 362
class uiPopupMenu(StructBase):
    block: lambda: POINTER(uiBlock)
    layout: c_void_p  # uiLayout
    but: lambda: POINTER(uiBut)
    butregion: lambda: POINTER(ARegion)

    if version > (3, 3, 1):
        title: c_char_p

    mxy: vec2i
    popup: c_bool
    slideout: c_bool
    # ... (cont)





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
    if version >= (5, 2, 0):
        rect: rcti
    else:
        icon: c_int

# # noinspection PyTypeHints
# # source/blender/makesdna/DNA_screen_types.h | rev 362
# class ARegion(StructBase):
#     next: lambda: POINTER(ARegion)
#     prev: lambda: POINTER(ARegion)
#
#     view2D: View2D
#     winrct: rcti
#     drawrct: rcti
#     winx: c_short
#     winy: c_short
#
#     if version > (3, 5):
#         category_scroll: c_int
#         _pad0: c_char * 4
#
#     visible: c_short
#     regiontype: c_short
#     alignment: c_short
#     flag: c_short
#
#     sizex: c_short
#     sizey: c_short
#
#     do_draw: c_short
#     do_draw_overlay: c_short  # (do_draw_paintcursor) they keep renaming this X_X
#     overlap: c_short
#     flagfullscreen: c_short
#
#     type: lambda: POINTER(ARegionType)  # ARegionType
#
#     uiblocks: ListBase(uiBlock)
#     panels: ListBase  # Panel
#     panels_category_active: ListBase
#     ui_lists: ListBase
#     ui_previews: ListBase
#     handlers: ListBase(wmEventHandler)
#     panels_category: ListBase
#
#     gizmo_map: c_void_p  # wmGizmoMap
#     regiontimer: c_void_p  # wmTimer
#     draw_buffer: c_void_p  # wmDrawBuffer
#
#     headerstr: c_char_p
#     regiondata: c_void_p
#
#     runtime: ARegion_Runtime

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

    runtime: c_void_p  # ARegion_Runtime
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
# source/blender/makesdna/DNA_screen_types.h | rev 362
class ScrArea_Runtime(StructBase):
    tool:           c_void_p  # bToolRef
    is_tool_set:    c_char
    _pad0:          c_char * 7


# # noinspection PyTypeHints
# class wmEvent(StructBase):
#     next: lambda: POINTER(wmEvent)
#     prev: lambda: POINTER(wmEvent)
#
#     # Event code itself (short, is also in key-map).
#     type: c_short
#     # Press, release, scroll-value.
#     val: c_short
#     # Mouse pointer position, screen coord.
#     xy: c_int * 2
#     # Region relative mouse position (name convention before Blender 2.5).
#     mval: c_int * 2
#     # A single UTF8 encoded character.
#     utf8_buf: c_char * 6
#     # Modifier states: #KM_SHIFT, #KM_CTRL, #KM_ALT, #KM_OSKEY & #KM_HYPER.
#     modifier: c_uint8
#     # The direction (for #KM_PRESS_DRAG events only).
#     direction: c_char
#     # Raw-key modifier (allow using any key as a modifier).
#     # Compatible with values in `type`.
#     keymodifier: c_short
#     # ...

# noinspection PyTypeHints
# source/blender/windowmanager/WM_types.h | rev 362
class wmEvent(StructBase):
    next: lambda: POINTER(wmEvent)
    prev: lambda: POINTER(wmEvent)

    type: c_short
    val: c_short

    if version < (3, 2):
        posx: c_short
        posy: c_short
        mvalx: c_short
        mvaly: c_short
    else:
        posx: c_int
        posy: c_int
        mvalx: c_int
        mvaly: c_int

    utf8_buf: c_char * 6

    if version < (3, 2, 2):
        ascii: c_char

    modifier: c_char

    # ... (cont)

    @property
    def ctrl(self) -> bool:
        return bool(int.from_bytes(self.modifier, "little") & 2)

    @property
    def shift(self) -> bool:
        return bool(int.from_bytes(self.modifier, "little") & 1)

    @property
    def alt(self) -> bool:
        return bool(int.from_bytes(self.modifier, "little") & 4)

    @property
    def type_string(self):
        return event_type_to_string(self.type)


# noinspection PyTypeHints
# source/blender/windowmanager/wm_event_system.h | rev 362
class wmEventHandler(StructBase):  # Generic
    next: lambda: POINTER(wmEventHandler)
    prev: lambda: POINTER(wmEventHandler)

    type: c_int  # enum eWM_EventHandlerType

    if version < (3, 5):
        flag: c_char

    if version >= (3, 5):
        flag: c_int  # enum eWM_EventHandlerFlag

    poll: c_void_p  # func EventHandlerPoll

# source/blender/blenkernel/BKE_context.h | rev 362
class bContextPollMsgDyn_Params(StructBase):
    get_fn: c_void_p
    free_fn: c_void_p
    user_data: c_void_p

# noinspection PyTypeHints
# makesdna\DNA_screen_types.h | rev 362
class bScreen(StructBase):
    id: lambda: ID
    vertbase: ListBase
    edgebase: ListBase
    areabase: ListBase
    regionbase: lambda: ListBase(ARegion)
    scene: c_void_p  # Scene, DNA_DEPRECATED
    flag: c_short
    winid: c_short
    redraws_flag: c_short
    temp: c_char
    state: c_char
    do_draw: c_char
    do_refresh: c_char
    do_draw_gesture: c_char
    do_draw_paintcursor: c_char
    do_draw_drag: c_char
    skip_handling: c_char
    scrubbing: c_char
    _pad1: c_char * 1
    active_region: lambda: POINTER(ARegion)
    animtimer: c_void_p  # wmTimer
    context: c_void_p
    tooltip: c_void_p  # wmTooltipState
    preview: c_void_p  # PreviewImage

# noinspection PyTypeHints
# source/blender/blenkernel/BKE_screen.h | rev 362
class SpaceType(StructBase):
    next: lambda: POINTER(SpaceType)
    prev: lambda: POINTER(SpaceType)

    name: c_char * 64  # BKE_ST_MAXNAME
    spaceid: c_int
    iconid: c_int

    create: c_void_p
    free: lambda: CFUNCTYPE(None, c_void_p)  # SpaceLink
    init: c_void_p
    exit: c_void_p
    listener: c_void_p

    deactivate: lambda: CFUNCTYPE(None, POINTER(ScrArea))
    refresh: c_void_p
    duplicate: c_void_p

    operatortypes: c_void_p
    keymap: c_void_p
    dropboxes: c_void_p

    gizmos: c_void_p
    context: c_void_p
    id_remap: c_void_p

    space_subtype_get: c_void_p
    space_subtype_set: c_void_p
    space_subtype_item_extend: c_void_p

    if version > (3, 3, 0):
        blend_read_data: c_void_p
        blend_read_lib: c_void_p
        blend_write: c_void_p

    regiontypes: lambda: ListBase(ARegionType)
    keymapflag: c_int


# noinspection PyTypeHints
# source/blender/blenkernel/BKE_screen.h | rev 362
class ARegionType(StructBase):
    next: lambda: POINTER(ARegionType)
    prev: lambda: POINTER(ARegionType)

    regionid: c_int

    init: c_void_p
    exit: c_void_p

    if version > (3, 5):
        poll: c_void_p

    draw: lambda: CFUNCTYPE(None, POINTER(bContext), POINTER(ARegion))

    if version > (2, 83):
        draw_overlay: c_void_p

    layout: c_void_p
    snap_size: c_void_p
    listener: lambda: CFUNCTYPE(None, c_void_p)
    message_subscribe: c_void_p

    free: c_void_p

    duplicate: c_void_p

    operatortypes: c_void_p
    keymap: c_void_p

    # Cursor handler
    cursor: lambda: CFUNCTYPE(None, POINTER(wmWindow), POINTER(ScrArea), POINTER(ARegion))

    context: c_void_p  # bContextDataCallback

    if version > (2, 83):
        on_view2d_changed: c_void_p

    drawcalls: ListBase
    paneltypes: ListBase
    headertypes: ListBase

    minsize: vec2i
    prefsize: vec2i
    keymapflag: c_int
    do_lock: c_short
    lock: c_short
    clip_gizmo_events_by_ui: c_bool
    event_cursor: c_short


# source/blender/makesdna/DNA_screen_types.h | rev 362
class ARegion_Runtime(StructBase):
    category: c_char_p

    visible_rect: rcti

    offset_x: c_int
    offset_y: c_int

    block_name_map: c_void_p  # GHash



# noinspection PyTypeHints
# source/blender/makesdna/DNA_space_types.h | rev 362
class SpaceLink(StructBase):
    next: lambda: POINTER(SpaceLink)
    prev: lambda: POINTER(SpaceLink)

    regionbase: ListBase(ARegion)
    spacetype: c_char
    link_flag: c_char
    _pad0: c_char * 6

# noinspection PyTypeHints
# source/blender/makesdna/DNA_screen_types.h | rev 362
class ScrArea(StructBase):
    next:                   lambda: POINTER(ScrArea)
    prev:                   lambda: POINTER(ScrArea)

    v1:                     c_void_p  # ScrVert
    v2:                     c_void_p  # ScrVert
    v3:                     c_void_p  # ScrVert
    v4:                     c_void_p  # ScrVert

    full:                   c_void_p  # bScreen
    totrct:                 rcti

    spacetype:              c_char
    butspacetype:           c_char
    butspacetype_subtype:   c_short

    win:                    vec2s
    headertype:             c_char  # DNA_DEPRECATED
    do_refresh:             c_char
    flag:                   c_short

    region_active_win:      c_short
    _pad2:                  c_char * 2

    type:                   POINTER(SpaceType)
    global_:                c_void_p  # ScrGlobalAreaData
    spacedata:              ListBase(SpaceLink)  # SpaceLink
    regionbase:             ListBase(ARegion)
    handlers:               lambda : ListBase(wmEventHandler)  # wmEventHandler and wmEventHandler_Op
    actionzones:            ListBase  # AZone
    runtime:                ScrArea_Runtime

    @property
    def action_zones(self):
        az = self.actionzones.first
        while az:
            yield az.contents
            az = az.contents.prev

# noinspection PyTypeHints
class wm(StructBase):
    manager: lambda: POINTER(wmWindowManager)
    window: lambda: POINTER(wmWindow)
    workspace: c_void_p  # WorkSpace
    screen: c_void_p  # bScreen
    area: lambda: POINTER(ScrArea)
    region: lambda: POINTER(ARegion)
    menu: lambda: POINTER(ARegion)
    gizmo_group: c_void_p  # wmGizmoGroup
    store: c_void_p  # bContextStore

    operator_poll_msg: c_char_p
    operator_poll_msg_dyn_params: bContextPollMsgDyn_Params

class bContext_data(StructBase):
    main: c_void_p  # Main
    scene: c_void_p  # Scene
    recursion: c_int
    py_init: c_bool
    py_context: c_void_p
    py_context_orig: c_void_p

# source/blender/blenkernel/intern/context.cc | rev 362
class bContext(StructBase):
    thread: c_int
    wm: wm
    data: bContext_data


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

    if version <= (4, 4, 3):
        addmousemove: c_char
    tag_cursor_refresh: c_char

    event_queue_check_click: c_char
    event_queue_check_drag: c_char
    event_queue_check_drag_handled: c_char

    if version < (3, 5, 0):
        _pad0: c_char * 1
    else:
        if version <= (4, 4, 3):
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
    if version <= (4, 4, 3):
        ime_data: c_void_p  # wmIMEData
    if version >= (4, 1, 0):
        if version <= (4, 4, 3):
            ime_data_is_composing: c_char
        else:
            addmousemove: c_char
        _pad1: c_char * 7

    if version <= (4, 4, 3):
        event_queue: ListBase
    handlers: ListBase(wmEventHandler)
    modalhandlers: lambda : ListBase(wmEventHandler)
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


# noinspection PyTypeHints
# source/blender/makesdna/DNA_windowmanager_types.h | rev 362
class wmOperator(StructBase):
    next:           lambda: POINTER(wmOperator)
    prev:           lambda: POINTER(wmOperator)

    idname:         c_char * 64  # OP_MAX_TYPENAME
    properties:     c_void_p     # IDProperty
    type:           lambda: POINTER(wmOperatorType)
    customdata:     c_void_p
    pyinstance:     c_void_p
    ptr:            lambda: POINTER(PointerRNA)
    reports:        c_void_p  # ReportList
    macro:          ListBase
    opm:            lambda: POINTER(wmOperator)
    layout:         c_void_p  # uiLayout
    flag:           c_short
    _pad6:          c_char * 6


# noinspection PyTypeHints
# source/blender/windowmanager/WM_types.h | rev 362
class wmOperatorType(StructBase):
    name:                   c_char_p
    idname:                 c_char_p
    translation_context:    c_char_p
    description:            c_char_p
    undo_group:             c_char_p

    exec:                   CFUNCTYPE(c_int, POINTER(bContext), POINTER(wmOperator))
    check:                  POINTER(c_bool)
    invoke:                 CFUNCTYPE(c_int, POINTER(bContext), POINTER(wmOperator), POINTER(wmEvent))
    cancel:                 CFUNCTYPE(None, POINTER(bContext), POINTER(wmOperator))
    modal:                  CFUNCTYPE(c_int, POINTER(bContext), POINTER(wmOperator), POINTER(wmEvent))
    poll:                   CFUNCTYPE(c_bool, POINTER(bContext))
    poll_property:          CFUNCTYPE(c_bool, POINTER(bContext), POINTER(wmOperator), c_void_p)  # PropertyRNA
    ui:                     CFUNCTYPE(None, POINTER(bContext), POINTER(wmOperator))
    get_name:               lambda: CFUNCTYPE(c_char_p, POINTER(wmOperatorType), POINTER(PointerRNA))
    get_description:        lambda: CFUNCTYPE(c_char_p, POINTER(bContext), POINTER(wmOperatorType), POINTER(PointerRNA))
    srna:                   c_void_p  # StructRNA

    last_properties:        c_void_p  # IDProperty
    prop:                   c_void_p  # PropertyRNA
    macro:                  ListBase  # wmOperatorTypeMacro
    modalkeymap:            c_void_p  # wmKeyMap
    pyop_poll:              lambda: CFUNCTYPE(c_bool, POINTER(bContext), POINTER(wmOperatorType))
    rna_ext:                c_void_p * 4  # ExtensionRNA

    if version > (2, 93):
        cursor_pending:     c_int

    flag:                   c_short

class context(StructBase):  # Anonymous
    # noinspection PyTypeHints
    win: lambda: POINTER(wmWindow)
    area: c_void_p  # ScrArea ptr
    region: c_void_p  # ARegion ptr
    region_type: c_short


# noinspection PyTypeHints
# source\blender\windowmanager\wm_event_system.h
class wmEventHandler_Op(StructBase):
    head: wmEventHandler
    op: lambda: POINTER(wmOperator)
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
    if version <= (3, 4, 1):
        name: c_char * 64
    else:
        name: c_char * 68
        _pad1: c_char * 4
    data: c_void_p

    # NOTE: This is beyond my understanding. These fields aren't affected at all, but because of it, everything breaks.
    # if version >= (3, 0, 0):
    #     anonymous_id: c_void_p

    if version >= (3, 6, 0):
        sharing_info: c_void_p


# noinspection PyTypeHints
class CustomData(StructBase):
    # NOTE: Adding CustomData layers to a bmesh will invalidate any existing pointers
    layers: lambda: POINTER(CustomDataLayer)


    if version <= (2, 82, 0):
        typemap: c_int * 42
        _pad1: c_char * 4
    elif version <= (2, 83, 20):
        typemap: c_int * 47
    elif version <= (2, 93, 9):
        typemap: c_int * 51
    elif version <= (3, 6, 23):
        typemap: c_int * 52
        _pad1: c_char * 4
    else:
        typemap: c_int * 53

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
    elem_table_dirty: c_char

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

    if version >= (5, 0, 0):
        use_toolflags: c_bool
    else:
        # if version >= (4, 2, 0) and version <= (4, 2, 1):
        #     use_toolflags: c_bool
        # else:
        use_toolflags: c_uint

    if version <= (2, 82, 0):
        currentop: c_void_p

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
