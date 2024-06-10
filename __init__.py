bl_info = {
    "name": "UniV",
    "description": "Advanced UV tools",
    "author": "Oxicid",
    "version": (0, 0, 1),
    "blender": (3, 2, 0),
    "category": "UV",
    "location": "N-panel in 2D and 3D view"
}

import bpy

# from bpy.types import Menu, Operator, Panel, PropertyGroup
# from bpy.props import (
#     StringProperty,
#     BoolProperty,
#     IntProperty,
#     IntVectorProperty,
#     FloatProperty,
#     FloatVectorProperty,
#     EnumProperty,
#     PointerProperty,
# )

from . import types
from . import preferences
from .types import bbox
from .types import btypes
from .types import island
from .utils import bench
from .utils import other
from .utils import text
from .utils import ubm
from .utils import umath
from .operators import transform
from . import ui


classes = (
    transform.UNIV_OT_Align,
    transform.UNIV_OT_Fill,
    transform.UNIV_OT_Crop,
    transform.UNIV_OT_Flip,
    transform.UNIV_OT_Rotate,
    transform.UNIV_OT_Sort,
    transform.UNIV_OT_Distribute,
    transform.UNIV_OT_Home,
    ui.UNIV_PT_General
)


def register():
    # Force reload by kaio: https://devtalk.blender.org/t/blender-2-91-addon-dev-workflow/15320/6
    from sys import modules
    from importlib import reload
    for _ in range(2):
        modules[__name__] = reload(modules[__name__])
        for name, module in modules.copy().items():
            if name.startswith(f"{__package__}."):
                globals()[name] = reload(module)

    for c in classes:
        bpy.utils.register_class(c)


def unregister():
    try:
        for c in reversed(classes):
            bpy.utils.unregister_class(c)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    register()
