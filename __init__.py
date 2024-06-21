bl_info = {
    "name": "UniV",
    "description": "Advanced UV tools",
    "author": "Oxicid",
    "version": (0, 1, 2),
    "blender": (3, 2, 0),
    "category": "UV",
    "location": "N-panel in 2D and 3D view"
}

import bpy
import traceback

from . import types        # noqa: F401
from . import preferences  # noqa: F401
from .types import bbox    # noqa: F401
from .types import btypes  # noqa: F401
from .types import island  # noqa: F401
from .utils import bench   # noqa: F401
from .utils import other   # noqa: F401
from .utils import text    # noqa: F401
from .utils import ubm     # noqa: F401
from .utils import umath   # noqa: F401
from .operators import transform
from .operators import toggle
from .operators import select
from . import ui
from . import keymaps
from . import preferences

classes = (
    preferences.UNIV_AddonPreferences,
    keymaps.UNIV_RestoreKeymaps,
    transform.UNIV_OT_Align,
    transform.UNIV_OT_Fill,
    transform.UNIV_OT_Crop,
    transform.UNIV_OT_Flip,
    transform.UNIV_OT_Rotate,
    transform.UNIV_OT_Sort,
    transform.UNIV_OT_Distribute,
    transform.UNIV_OT_Home,
    toggle.UNIV_OT_SplitUVToggle,
    toggle.UNIV_OT_SyncUVToggle,
    select.UNIV_OT_SelectLinked,
    ui.UNIV_PT_General
)


def register():
    # Force reload by kaio: https://devtalk.blender.org/t/blender-2-91-addon-dev-workflow/15320/6
    from sys import modules
    from importlib import reload
    modules[__name__] = reload(modules[__name__])
    for name, module in modules.copy().items():
        if name.startswith(f"{__package__}."):
            globals()[name] = reload(module)

    for c in classes:
        bpy.utils.register_class(c)

    bpy.types.VIEW3D_HT_header.prepend(toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.prepend(toggle.univ_header_sync_btn)
    bpy.types.IMAGE_HT_header.prepend(toggle.univ_header_split_btn)

    try:
        keymaps.add_keymaps()
    except AttributeError:  # noqa
        traceback.print_exc()


def unregister():
    keymaps.remove_keymaps()

    try:
        for c in reversed(classes):
            bpy.utils.unregister_class(c)
    except Exception:  # noqa
        traceback.print_exc()

    bpy.types.VIEW3D_HT_header.remove(toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.remove(toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.remove(toggle.univ_header_sync_btn)


if __name__ == "__main__":
    register()
