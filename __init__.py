# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "UniV",
    "description": "Advanced UV tools",
    "author": "Oxicid",
    "version": (0, 8, 1),
    "blender": (3, 2, 0),
    "category": "UV",
    "location": "N-panel in 2D and 3D view"
}

import bpy
import traceback

from . import types        # noqa: F401
from . import preferences  # noqa: F401
from .utils import bench   # noqa: F401
from .utils import other   # noqa: F401
from .utils import text    # noqa: F401
from .utils import ubm     # noqa: F401
from .utils import umath   # noqa: F401
from .types import bbox    # noqa: F401
from .types import btypes  # noqa: F401
from .types import island  # noqa: F401
from .types import mesh_island  # noqa: F401
from .operators import straight
from .operators import quadrify
from .operators import relax
from .operators import unwrap
from .operators import transform
from .operators import toggle
from .operators import select
from .operators import seam
from .operators import quick_snap
from . import ui
from . import keymaps
from . import preferences

classes = (
    preferences.UNIV_AddonPreferences,
    keymaps.UNIV_RestoreKeymaps,
    # Transforms
    transform.UNIV_OT_Orient,
    transform.UNIV_OT_Align,
    transform.UNIV_OT_Fill,
    transform.UNIV_OT_Crop,
    transform.UNIV_OT_Flip,
    transform.UNIV_OT_Rotate,
    transform.UNIV_OT_Sort,
    transform.UNIV_OT_Distribute,
    transform.UNIV_OT_Home,
    transform.UNIV_OT_Random,
    transform.UNIV_OT_Weld,
    transform.UNIV_OT_Stitch,
    # Quadrify
    quadrify.UNIV_OT_Quad,
    straight.UNIV_OT_Straight,
    relax.UNIV_OT_Relax,
    unwrap.UNIV_OT_Unwrap,
    # Toggles
    toggle.UNIV_OT_SplitUVToggle,
    toggle.UNIV_OT_SyncUVToggle,
    # Selects
    select.UNIV_OT_SelectLinked,
    select.UNIV_OT_Select_By_Cursor,
    select.UNIV_OT_SelectView,
    select.UNIV_OT_Single,
    select.UNIV_OT_Select_Square_Island,
    select.UNIV_OT_Select_Border,
    select.UNIV_OT_Select_Inner,
    select.UNIV_OT_Select_Zero,
    select.UNIV_OT_Select_Flipped,
    select.UNIV_OT_Select_Border_Edge_by_Angle,
    # QuickSnap
    quick_snap.UNIV_OT_QuickSnap,
    # UI
    ui.UNIV_PT_General,
    ui.UNIV_PT_General_VIEW_3D,
    # Seam
    seam.UNIV_OT_Cut_VIEW2D,
    seam.UNIV_OT_Cut_VIEW3D,
    seam.UNIV_OT_Angle,
)

is_enabled = False

def register():
    # Force reload by kaio: https://devtalk.blender.org/t/blender-2-91-addon-dev-workflow/15320/6
    global is_enabled
    if is_enabled:
        import sys
        import importlib
        sys.modules[__name__] = importlib.reload(sys.modules[__name__])
        for name, module in sys.modules.copy().items():
            if name.startswith(f"{__package__}."):
                globals()[name] = importlib.reload(module)
    is_enabled = True

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
