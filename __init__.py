# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "UniV",
    "description": "Advanced UV tools",
    "author": "Oxicid",
    "version": (2, 7, 4),
    "blender": (3, 2, 0),
    "category": "UV",
    "location": "N-panel in 2D and 3D view"
}

import bpy
import traceback

from . import types        # noqa: F401 # pylint:disable=unused-import
from . import preferences  # noqa: F401 # pylint:disable=unused-import
from .utils import bench   # noqa: F401 # pylint:disable=unused-import
from .utils import other   # noqa: F401 # pylint:disable=unused-import
from .utils import ubm     # noqa: F401 # pylint:disable=unused-import
from .utils import umath   # noqa: F401 # pylint:disable=unused-import
from .types import bbox    # noqa: F401 # pylint:disable=unused-import
from .types import btypes  # noqa: F401 # pylint:disable=unused-import
from .types import island  # noqa: F401 # pylint:disable=unused-import
from .types import mesh_island  # noqa: F401 # pylint:disable=unused-import
from .operators import checker
from .operators import inspect
from .operators import misc
from .operators import project
from .operators import quadrify
from .operators import quick_snap
from .operators import relax
from .operators import seam
from .operators import select
from .operators import stack
from .operators import straight
from .operators import toggle
from .operators import transform
from .operators import unwrap
from . import ui
from . import keymaps
from . import preferences

try:
    classes = (
        preferences.UNIV_AddonPreferences,
        preferences.UNIV_Settings,
        keymaps.UNIV_RestoreKeymaps,
        # Checker System
        checker.UNIV_OT_Checker,
        checker.UNIV_OT_CheckerCleanup,
        # Inspect
        inspect.UNIV_OT_Check_Zero,
        inspect.UNIV_OT_Check_Flipped,
        inspect.UNIV_OT_Check_Non_Splitted,
        inspect.UNIV_OT_Check_Overlap,
        # Transforms
        transform.UNIV_OT_Orient,
        transform.UNIV_OT_Orient_VIEW3D,
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
        transform.UNIV_OT_AdjustScale,
        transform.UNIV_OT_AdjustScale_VIEW3D,
        transform.UNIV_OT_Normalize,
        transform.UNIV_OT_Normalize_VIEW3D,
        transform.UNIV_OT_Pack,
        # Quadrify
        quadrify.UNIV_OT_Quadrify,
        straight.UNIV_OT_Straight,
        relax.UNIV_OT_Relax,
        unwrap.UNIV_OT_Unwrap,
        # Toggles
        toggle.UNIV_OT_SplitUVToggle,
        toggle.UNIV_OT_SyncUVToggle,
        # Selects
        select.UNIV_OT_SelectLinked,
        select.UNIV_OT_Select_By_Cursor,
        select.UNIV_OT_Select_Square_Island,
        select.UNIV_OT_Select_Border,
        select.UNIV_OT_Select_Border_Edge_by_Angle,
        select.UNIV_OT_Select_Pick,
        select.UNIV_OT_Select_Grow,
        select.UNIV_OT_Select_Edge_Grow_VIEW2D,
        select.UNIV_OT_Tests,
        # QuickSnap
        quick_snap.UNIV_OT_QuickSnap,
        # UI
        ui.UNIV_PT_General,
        ui.UNIV_PT_General_VIEW_3D,
        ui.UNIV_PT_PackSettings,
        # Seam
        seam.UNIV_OT_Cut_VIEW2D,
        seam.UNIV_OT_Cut_VIEW3D,
        seam.UNIV_OT_Angle,
        seam.UNIV_OT_SeamBorder,
        # Project
        project.UNIV_Normal,
        project.UNIV_BoxProject,
        # Stack
        stack.UNIV_OT_Stack,
        stack.UNIV_OT_Stack_VIEW3D,
        # Misc
        misc.UNIV_OT_Pin,
    )
except AttributeError:
    traceback.print_exc()
    classes = ()

is_enabled = False

def register():
    global is_enabled
    if is_enabled or not classes:
        from . import reload
        reload.reload(globals())
        if not classes:
            raise AttributeError('Failed to load operators, try re-enabling or restarting Blender')
    is_enabled = True

    for c in classes:
        try:
            bpy.utils.register_class(c)
        except Exception:  # noqa
            traceback.print_exc()

    bpy.types.Scene.univ_settings = bpy.props.PointerProperty(type=preferences.UNIV_Settings)
    bpy.types.VIEW3D_HT_header.prepend(toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.prepend(toggle.univ_header_sync_btn)
    bpy.types.IMAGE_HT_header.prepend(toggle.univ_header_split_btn)

    try:
        keymaps.add_keymaps()
    except AttributeError:
        traceback.print_exc()


def unregister():
    keymaps.remove_keymaps()

    for c in reversed(classes):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            if not hasattr(c, 'rna_type'):
                continue
            traceback.print_exc()

    bpy.types.VIEW3D_HT_header.remove(toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.remove(toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.remove(toggle.univ_header_sync_btn)

    for scene in bpy.data.scenes:
        if "univ_settings" in scene:
            del scene["univ_settings"]


if __name__ == "__main__":
    register()
