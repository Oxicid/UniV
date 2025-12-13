# SPDX-FileCopyrightText: 2025 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "UniV",
    "description": "Advanced UV tools",
    "author": "Oxicid",
    "version": (3, 9, 37),
    "blender": (3, 2, 0),
    "category": "UV",
    "location": "N-panel in 2D and 3D view"
}

import bpy
import typing
import traceback

from . import utypes        # noqa: F401 # pylint:disable=unused-import
from . import preferences  # noqa: F401 # pylint:disable=unused-import
from .utils import bench   # noqa: F401 # pylint:disable=unused-import
from .utils import other   # noqa: F401 # pylint:disable=unused-import
from .utils import ubm     # noqa: F401 # pylint:disable=unused-import
from .utils import umath   # noqa: F401 # pylint:disable=unused-import
from .utypes import bbox    # noqa: F401 # pylint:disable=unused-import
from .utypes import btypes  # noqa: F401 # pylint:disable=unused-import
from .utypes import island  # noqa: F401 # pylint:disable=unused-import
from .utypes import mesh_island  # noqa: F401 # pylint:disable=unused-import
from .operators import checker
from .operators import inspect
from .operators import misc
from .operators import project
from .operators import quadrify
from .operators import quick_snap
from .operators import relax
from .operators import seam
from .operators import select
from .operators import stitch_and_weld
from .operators import stack
from .operators import straight
from .operators import symmetrize
from .operators import texel
from .operators import toggle
from .operators import transform
from .operators import unwrap
from . import ui
from . import draw
from . import icons
from . import keymaps
from . import fastapi
from . import preferences

from bpy.app.timers import register as tm_register

univ_pro: "type[bpy?] | None"
try:
    from . import univ_pro
except ImportError:
    univ_pro = None

classes: list[bpy.types.Operator | bpy.types.Panel | bpy.types.Macro | typing.Any] = []
classes_workspace = []


def load_register_types():
    global classes
    try:
        classes.extend([
            preferences.UNIV_UV_Layers,
            preferences.UNIV_TrimPreset,
            preferences.UNIV_TexelPreset,
            preferences.UNIV_AddonPreferences,
            preferences.UNIV_OT_ShowAddonPreferences,
            keymaps.UNIV_RestoreKeymaps,
            # Checker System
            checker.UNIV_OT_Checker,
            checker.UNIV_OT_CheckerCleanup,
            # Inspect
            inspect.UNIV_OT_BatchInspectFlags,
            inspect.UNIV_OT_BatchInspect,
            inspect.UNIV_OT_Check_Zero,
            inspect.UNIV_OT_Check_Flipped,
            inspect.UNIV_OT_Check_Non_Splitted,
            inspect.UNIV_OT_Check_Overlap,
            inspect.UNIV_OT_Check_Over,
            inspect.UNIV_OT_Check_Other,
            # Stitch and Weld
            stitch_and_weld.UNIV_OT_Weld_VIEW3D,
            stitch_and_weld.UNIV_OT_Weld,
            stitch_and_weld.UNIV_OT_Stitch_VIEW3D,
            stitch_and_weld.UNIV_OT_Stitch,
            # Transforms
            transform.UNIV_OT_Orient_VIEW3D,
            transform.UNIV_OT_Orient,
            transform.UNIV_OT_Gravity,
            transform.UNIV_OT_Align,
            transform.UNIV_OT_Align_pie,
            transform.UNIV_OT_Fill,
            transform.UNIV_OT_Crop,
            transform.UNIV_OT_Flip,
            transform.UNIV_OT_Flip_VIEW3D,
            transform.UNIV_OT_Rotate,
            transform.UNIV_OT_Rotate_VIEW3D,
            transform.UNIV_OT_Sort,
            transform.UNIV_OT_Distribute,
            transform.UNIV_OT_Home_VIEW3D,
            transform.UNIV_OT_Home,
            transform.UNIV_OT_Shift_VIEW3D,
            transform.UNIV_OT_Shift,
            transform.UNIV_OT_Random_VIEW3D,
            transform.UNIV_OT_Random,
            transform.UNIV_OT_Pack,
            # Texel
            texel.UNIV_OT_ResetScale_VIEW3D,
            texel.UNIV_OT_ResetScale,
            texel.UNIV_OT_AdjustScale,
            texel.UNIV_OT_AdjustScale_VIEW3D,
            texel.UNIV_OT_Normalize,
            texel.UNIV_OT_Normalize_VIEW3D,
            texel.UNIV_OT_TexelDensitySet,
            texel.UNIV_OT_TexelDensitySet_VIEW3D,
            texel.UNIV_OT_TexelDensityGet,
            texel.UNIV_OT_TexelDensityGet_VIEW3D,
            texel.UNIV_OT_TexelDensityFromTexture,
            texel.UNIV_OT_TexelDensityFromPhysicalSize,
            texel.UNIV_OT_CalcUDIMsFrom_3DArea_VIEW3D,
            texel.UNIV_OT_CalcUDIMsFrom_3DArea,
            texel.UNIV_OT_Calc_UV_Area_VIEW3D,
            texel.UNIV_OT_Calc_UV_Area,
            texel.UNIV_OT_Calc_UV_Coverage_VIEW3D,
            texel.UNIV_OT_Calc_UV_Coverage,
            # Symmetrize
            symmetrize.UNIV_OT_Symmetrize,
            # Quadrify
            quadrify.UNIV_OT_Quadrify,
            # Straight
            straight.UNIV_OT_Straight,
            # Relax
            relax.UNIV_OT_Relax,
            relax.UNIV_OT_Relax_VIEW3D,
            # Unwrap
            unwrap.UNIV_OT_Unwrap_VIEW3D,
            # Toggles
            toggle.UNIV_OT_SplitUVToggle,
            toggle.UNIV_OT_TogglePivot,
            toggle.UNIV_OT_TogglePanelsByCursor,
            toggle.UNIV_OT_SyncUVToggle,
            toggle.UNIV_OT_StretchUVToggle,
            toggle.UNIV_OT_ShowModifiedUVEdgeToggle,
            toggle.UNIV_OT_WorkspaceToggle,
            # Modifier Toggle
            toggle.UNIV_OT_ModifiersToggle,

            # Selects
            select.UNIV_OT_SelectLinked,
            select.UNIV_OT_SelectLinked_VIEW3D,
            select.UNIV_OT_Select_By_Cursor,
            select.UNIV_OT_Select_Square_Island,
            select.UNIV_OT_Select_Border,
            select.UNIV_OT_Select_Pick,
            select.UNIV_OT_SelectLinkedPick_VIEW3D,
            select.UNIV_OT_DeselectLinkedPick_VIEW3D,
            select.UNIV_OT_Select_Grow_VIEW3D,
            select.UNIV_OT_Select_Grow,
            select.UNIV_OT_Select_Edge_Grow_VIEW2D,
            select.UNIV_OT_Select_Edge_Grow_VIEW3D,
            select.UNIV_OT_SelectTexelDensity,
            select.UNIV_OT_SelectTexelDensity_VIEW3D,
            select.UNIV_OT_SelectByArea,
            select.UNIV_OT_Stacked,
            select.UNIV_OT_SelectByVertexCount_VIEW2D,
            select.UNIV_OT_SelectByVertexCount_VIEW3D,
            select.UNIV_OT_SelectMode,
            select.UNIV_OT_Tests,
            # QuickSnap
            quick_snap.UNIV_OT_QuickSnap,
            # UI
            ui.UNIV_PT_TD_LayersManager,
            ui.UNIV_UL_TD_PresetsManager,
            ui.UNIV_PT_TD_PresetsManager,
            ui.UNIV_PT_TD_PresetsManager_VIEW3D,
            ui.UNIV_UL_UV_LayersManager,
            ui.UNIV_UL_UV_LayersManagerV2,
            ui.UNIV_PT_General_VIEW_3D,
            ui.UNIV_PT_General,
            ui.UNIV_PT_GlobalSettings,
            ui.UNIV_PT_PackSettings,
            ui.UNIV_PT_BatchInspectSettings,
            # Pie Menus
            ui.IMAGE_MT_PIE_univ_inspect,
            ui.IMAGE_MT_PIE_univ_align,
            ui.IMAGE_MT_PIE_univ_misc,
            ui.VIEW3D_MT_PIE_univ_misc,
            ui.VIEW3D_MT_PIE_univ_obj,
            ui.IMAGE_MT_PIE_univ_edit,
            ui.VIEW3D_MT_PIE_univ_edit,
            ui.VIEW3D_MT_PIE_univ_favorites_edit,
            ui.IMAGE_MT_PIE_univ_favorites_edit,
            ui.VIEW3D_MT_PIE_univ_projection,
            ui.IMAGE_MT_PIE_univ_texel,
            ui.VIEW3D_MT_PIE_univ_texel,
            ui.IMAGE_MT_PIE_univ_transform,
            # Icons Generator
            icons.UNIV_OT_IconsGenerator,
            # Seam
            seam.UNIV_OT_Cut_VIEW2D,
            seam.UNIV_OT_Cut_VIEW3D,
            seam.UNIV_OT_Angle,
            seam.UNIV_OT_SeamBorder,
            seam.UNIV_OT_SeamBorder_VIEW3D,
            # Project
            project.UNIV_OT_Normal,
            project.UNIV_OT_ViewProject,
            project.UNIV_OT_SmartProject,
            # Misc
            misc.UNIV_OT_Pin,
            misc.UNIV_OT_Hide,
            misc.UNIV_OT_Focus,
            misc.UNIV_OT_SetCursor2D,
            misc.UNIV_OT_TD_PresetsProcessing,
            misc.UNIV_OT_FixUVs,
            misc.UNIV_OT_Join,
            misc.UNIV_OT_Add,
            misc.UNIV_OT_Remove,
            misc.UNIV_OT_MoveUp,
            misc.UNIV_OT_MoveDown,
            misc.UNIV_OT_CopyToLayer,
            misc.UNIV_OT_SetActiveRender,
            # Mesh
            misc.UNIV_OT_Flatten,
            misc.UNIV_OT_FlattenCleanup,
        ])

        if univ_pro:
            classes.extend((
                # UI
                ui.UNIV_UL_TrimPresetsManager,
                # Trim
                univ_pro.trim.UNIV_OT_TD_PresetsProcessing,
                # Stack
                univ_pro.stack.UNIV_OT_Stack,
                univ_pro.stack.UNIV_OT_Stack_VIEW3D,
                # Select
                univ_pro.select.UNIV_OT_Select_Flat_VIEW3D,
                univ_pro.select.UNIV_OT_Select_Flat,
                univ_pro.select.UNIV_OT_SelectSimilar_VIEW2D,
                univ_pro.select.UNIV_OT_SelectSimilar_VIEW3D,

                univ_pro.select.UNIV_OT_Select_Loop_Pick_VIEW3D,
                univ_pro.select.UNIV_OT_Select_Loop_VIEW3D,
                univ_pro.select.UNIV_OT_Select_Loop_VIEW2D,
                # Transform
                univ_pro.drag.UNIV_OT_Drag,
                # Transfer
                univ_pro.transfer.UNIV_OT_Transfer,
                # Misc
                univ_pro.rectify.UNIV_OT_Rectify,
                univ_pro.projection.UNIV_OT_BoxProject,
                # Unwrap
                univ_pro.unwrap.UNIV_OT_Unwrap,

            ))

        else:
            classes.extend((
                # Project
                project.UNIV_OT_BoxProject,
                # Stack
                stack.UNIV_OT_Stack,
                stack.UNIV_OT_Stack_VIEW3D,
                # Unwrap
                unwrap.UNIV_OT_Unwrap
            ))
    except AttributeError:
        traceback.print_exc()
        classes = []


load_register_types()


def load_workspace_types():
    classes_workspace.extend([
        ui.UNIV_WT_edit_VIEW3D,
        ui.UNIV_WT_object_VIEW3D]
    )


load_workspace_types()

is_enabled = False


def register():
    if bpy.app.background:
        print("UniV: Skipping registration in background mode")
        return

    global is_enabled
    if is_enabled or not classes:
        from . import reload
        reload.reload(globals())

        if not classes:
            load_register_types()
            if not classes:
                raise AttributeError('UniV: Failed to load operators, try re-enabling or restarting Blender')

        if not classes_workspace:
            load_workspace_types()
            if not classes_workspace:
                raise AttributeError('UniV: Failed to load workspace tool, try re-enabling or restarting Blender')

    is_enabled = True

    for c in classes:
        try:
            bpy.utils.register_class(c)

            # Register Macros
            if c.__name__ == 'UNIV_OT_SelectLinkedPick_VIEW3D':
                item = c.define("MESH_OT_select_linked_pick")
                item.properties.deselect = False
            elif c.__name__ == 'UNIV_OT_DeselectLinkedPick_VIEW3D':
                item = c.define("MESH_OT_select_linked_pick")
                item.properties.deselect = True

        except Exception:  # noqa
            print(f'UniV: Failed to register a class {c.__name__}')
            traceback.print_exc()

    for c in classes_workspace:
        try:
            bpy.utils.register_tool(c)
        except Exception:  # noqa
            print(f'UniV: Failed to register a class {c.__name__}')
            traceback.print_exc()

    # WARNING: When modules are reloaded, classes are overwritten and have no registration.
    # To avoid this, it is necessary to use initially registered classes.
    # Perhaps it does not allow to reload operators in a normal way.
    # bpy.types.WindowManager.univ_settings = bpy.props.PointerProperty(type=classes[3])

    tm_register(draw.shaders.Shaders.init_shaders, first_interval=0.09, persistent=True)
    tm_register(misc.UNIV_OT_UV_Layers_Manager.append_handler_with_delay, first_interval=0.1, persistent=True)
    if univ_pro:
        tm_register(draw.DrawerSubscribeRNA.register_handler, first_interval=0.1, persistent=True)
        tm_register(draw.DrawerSubscribeRNA.subscribe, first_interval=0.15)  # NOTE: Call after register_handler
        tm_register(draw.Drawer2D.append_handler_with_delay, first_interval=0.1, persistent=True)
        tm_register(draw.Drawer3D.append_handler_with_delay, first_interval=0.1, persistent=True)
        tm_register(draw.TrimDrawer.append_handler_with_delay, first_interval=0.1, persistent=True)

    bpy.types.VIEW3D_HT_header.prepend(toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.prepend(toggle.univ_header_sync_btn)
    bpy.types.IMAGE_HT_header.prepend(toggle.univ_header_split_btn)

    try:
        fastapi.clib.FastAPI.load()
    except:  # noqa
        pass
        # print('UniV: Cannot load fastapi')

    try:
        icons.icons.register_icons_()
    except:  # noqa
        print('UniV: Icons not loaded')
        traceback.print_exc()

    try:
        keymaps.add_keymaps()
        keymaps.add_keymaps_ws()
    except AttributeError:
        traceback.print_exc()

    preferences.update_panel(None, None)
    toggle.ToggleHandlers.register_handler()


def unregister():
    if bpy.app.background:
        return

    keymaps.remove_keymaps()
    keymaps.remove_keymaps_ws()
    icons.icons.unregister_icons_()

    draw.DrawerSubscribeRNA.unregister_handler()
    draw.Drawer2D.unregister()
    draw.Drawer3D.unregister()
    draw.TrimDrawer.unregister()

    for handle in reversed(bpy.app.handlers.depsgraph_update_post):
        if handle.__name__.startswith('univ_'):
            bpy.app.handlers.depsgraph_update_post.remove(handle)

    # del bpy.types.WindowManager.univ_settings  # noqa
    # for scene in bpy.data.scenes:
    #     if "univ_settings" in scene:
    #         del scene["univ_settings"]

    for c in reversed(classes_workspace):
        try:
            bpy.utils.unregister_tool(c)
        except RuntimeError:
            # if not hasattr(c, 'rna_type'):
            #     continue
            traceback.print_exc()

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
    texel.UNIV_OT_TexelDensityFromTexture.store_poliigon_physical_size_cache()

    toggle.ToggleHandlers.unregister_handler()


if __name__ == "__main__":
    register()
