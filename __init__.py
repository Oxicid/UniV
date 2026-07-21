# SPDX-FileCopyrightText: 2026 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "UniV",
    "description": "Smart UV tools",
    "author": "Oxicid",
    "version": (4, 0, 5),
    "blender": (3, 2, 0),
    "category": "UV",
    "location": "N-panel in 2D and 3D view"
}

import bpy
import typing
import traceback

from importlib.util import find_spec
univ_pro_exist = find_spec(f"{__package__}.univ_pro") is not None
del find_spec

if "NOT_BL_EXT":
    from . import fastapi

classes: list[bpy.types.Operator | bpy.types.Panel | bpy.types.Macro | typing.Any] = []
classes_workspace = []


def load_register_types():
    global classes

    from . import preferences
    from . import operators
    from . import ui
    from . import icons
    from . import keymaps

    try:
        classes.extend([
            preferences.UNIV_UV_Layers,
            preferences.UNIV_TrimPreset,
            preferences.UNIV_TrimPresetsSlot,
            preferences.UNIV_TexelPreset,
            preferences.UNIV_AddonPreferences,
            preferences.UNIV_OT_ShowAddonPreferences,
            keymaps.UNIV_RestoreKeymaps,
            # Checker System
            operators.checker.UNIV_OT_CheckerCleanup,
            # Inspect
            operators.inspect.UNIV_OT_BatchInspectFlags,
            operators.inspect.UNIV_OT_BatchInspect,
            operators.inspect.UNIV_OT_Check_Zero,
            operators.inspect.UNIV_OT_Check_Flipped,
            operators.inspect.UNIV_OT_Check_Non_Splitted,
            operators.inspect.UNIV_OT_Check_Overlap,
            operators.inspect.UNIV_OT_Check_Over,
            operators.inspect.UNIV_OT_Check_Other,
            operators.inspect.UNIV_OT_Check_Lib,
            # Stitch and Weld
            operators.stitch_and_weld.UNIV_OT_Weld_VIEW3D,
            operators.stitch_and_weld.UNIV_OT_Weld,
            operators.stitch_and_weld.UNIV_OT_Stitch_VIEW3D,
            operators.stitch_and_weld.UNIV_OT_Stitch,
            # Transforms
            operators.transform.UNIV_OT_Orient_VIEW3D,
            operators.transform.UNIV_OT_Orient,
            operators.transform.UNIV_OT_Gravity_VIEW3D,
            operators.transform.UNIV_OT_Gravity_VIEW2D,
            operators.transform.UNIV_OT_Align,
            operators.transform.UNIV_OT_Align_pie,
            operators.transform.UNIV_OT_Fill,
            operators.transform.UNIV_OT_Fit,
            operators.transform.UNIV_OT_SnapToPixels,
            operators.transform.UNIV_OT_Flip,
            operators.transform.UNIV_OT_Flip_VIEW3D,
            operators.transform.UNIV_OT_Rotate,
            operators.transform.UNIV_OT_Rotate_VIEW3D,
            operators.transform.UNIV_OT_Sort,
            operators.transform.UNIV_OT_Distribute,
            operators.transform.UNIV_OT_Break,
            operators.transform.UNIV_OT_Home_VIEW3D,
            operators.transform.UNIV_OT_Home,
            operators.transform.UNIV_OT_Shift_VIEW3D,
            operators.transform.UNIV_OT_Shift,
            operators.transform.UNIV_OT_Random_VIEW3D,
            operators.transform.UNIV_OT_Random,
            operators.transform.UNIV_OT_PackOther,
            operators.transform.UNIV_OT_Pack,
            # Texel
            operators.texel.UNIV_OT_ResetScale_VIEW3D,
            operators.texel.UNIV_OT_ResetScale,
            operators.texel.UNIV_OT_AdjustScale,
            operators.texel.UNIV_OT_AdjustScale_VIEW3D,
            operators.texel.UNIV_OT_Normalize,
            operators.texel.UNIV_OT_Normalize_VIEW3D,
            operators.texel.UNIV_OT_TexelDensitySet,
            operators.texel.UNIV_OT_TexelDensitySet_VIEW3D,
            operators.texel.UNIV_OT_TexelDensityGet,
            operators.texel.UNIV_OT_TexelDensityGet_VIEW3D,
            operators.texel.UNIV_OT_TexelDensityFromPhysicalSize,
            operators.texel.UNIV_OT_CalcUDIMsFrom_3DArea_VIEW3D,
            operators.texel.UNIV_OT_CalcUDIMsFrom_3DArea,
            operators.texel.UNIV_OT_Calc_UV_Area_VIEW3D,
            operators.texel.UNIV_OT_Calc_UV_Area,
            operators.texel.UNIV_OT_Calc_UV_Coverage_VIEW3D,
            operators.texel.UNIV_OT_Calc_UV_Coverage,
            # Symmetrize
            operators.symmetrize.UNIV_OT_Symmetrize,
            # Quadrify
            operators.quadrify.UNIV_OT_Quadrify,
            # Relax
            operators.relax.UNIV_OT_Relax,
            operators.relax.UNIV_OT_Relax_VIEW3D,
            # Unwrap
            operators.unwrap.UNIV_OT_Unwrap_VIEW3D,
            # Toggles
            operators.toggle.UNIV_OT_SplitUVToggle,
            operators.toggle.UNIV_OT_TogglePivot,
            operators.toggle.UNIV_OT_TogglePanelsByCursor,
            operators.toggle.UNIV_OT_SyncUVToggle,
            operators.toggle.UNIV_OT_StretchUVToggle,
            operators.toggle.UNIV_OT_ShowModifiedUVEdgeToggle,
            operators.toggle.UNIV_OT_WorkspaceToggle,
            # Modifier Toggle
            operators.toggle.UNIV_OT_ModifiersToggle,

            # Selects
            operators.select.UNIV_OT_SelectLinked,
            operators.select.UNIV_OT_SelectLinked_VIEW3D,
            operators.select.UNIV_OT_Select_By_Cursor,
            operators.select.UNIV_OT_Select_Square_Island,
            operators.select.UNIV_OT_Select_Border,
            operators.select.UNIV_OT_Select_Pick,
            operators.select.UNIV_OT_SelectLinkedPick_VIEW3D,
            operators.select.UNIV_OT_DeselectLinkedPick_VIEW3D,
            operators.select.UNIV_OT_Select_Grow_VIEW3D,
            operators.select.UNIV_OT_Select_Grow,
            operators.select.UNIV_OT_Select_Edge_Grow_VIEW2D,
            operators.select.UNIV_OT_Select_Edge_Grow_VIEW3D,
            operators.select.UNIV_OT_SelectTexelDensity,
            operators.select.UNIV_OT_SelectTexelDensity_VIEW3D,
            operators.select.UNIV_OT_SelectByArea,
            operators.select.UNIV_OT_Stacked,
            operators.select.UNIV_OT_SelectByVertexCount_VIEW2D,
            operators.select.UNIV_OT_SelectByVertexCount_VIEW3D,
            operators.select.UNIV_OT_SelectMode,
            operators.select.UNIV_OT_LocalInvertSelection,
            operators.select.UNIV_OT_Tests,
            # QuickSnap
            operators.quick_snap.UNIV_OT_QuickSnap,
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
            operators.mark.UNIV_OT_Pin,
            operators.mark.UNIV_OT_Mark_VIEW2D,
            operators.mark.UNIV_OT_Mark_VIEW3D,
            operators.mark.UNIV_OT_Cut_VIEW2D,
            operators.mark.UNIV_OT_Cut_VIEW3D,
            operators.mark.UNIV_OT_Angle,
            operators.mark.UNIV_OT_SeamBorder,
            operators.mark.UNIV_OT_SeamBorder_VIEW3D,
            operators.mark.UNIV_OT_SeamBorderSimple_VIEW2D,
            operators.mark.UNIV_OT_SeamBorderSimple_VIEW3D,
            # Project
            operators.project.UNIV_OT_Normal,
            operators.project.UNIV_OT_ViewProject,
            operators.project.UNIV_OT_SmartProject,
            operators.project.UNIV_OT_Flatten,
            operators.project.UNIV_OT_FlattenCleanup,
            operators.project.UNIV_OT_WrapProject,
            # Misc
            operators.misc.UNIV_OT_RandomColor,
            operators.misc.UNIV_OT_LinearGradient,
            operators.misc.UNIV_OT_Hide,
            operators.misc.UNIV_OT_Focus,
            operators.misc.UNIV_OT_SetCursor2D,
            operators.misc.UNIV_OT_TD_PresetsProcessing,
            operators.misc.UNIV_OT_FixUVs,
            operators.misc.UNIV_OT_Join,
            operators.misc.UNIV_OT_Add,
            operators.misc.UNIV_OT_Remove,
            operators.misc.UNIV_OT_MoveUp,
            operators.misc.UNIV_OT_MoveDown,
            operators.misc.UNIV_OT_CopyToLayer,
            operators.misc.UNIV_OT_SetActiveRender,
            operators.misc.UNIV_OT_SmartScaleApply,
            operators.misc.UNIV_OT_AlignBorderVerts,
        ])

        if univ_pro_exist:
            from . import univ_pro
            classes.extend((
                # UI
                ui.UNIV_UL_TrimPresetsManager,
                ui.UNIV_UL_TrimSlotsManager,
                ui.UNIV_PT_TrimManager,
                ui.UNIV_PT_CheckerSettings,
                ui.UNIV_PT_CheckerTextures,

                # Checker System
                univ_pro.checker.UNIV_OT_Checker,
                univ_pro.checker.UNIV_OT_CheckerSave,
                univ_pro.checker.UNIV_OT_CheckerUpdate,
                univ_pro.checker.UNIV_OT_CheckerShowFolder,
                univ_pro.checker.UNIV_OT_CheckerGenerator,
                # Trim
                univ_pro.trim.UNIV_OT_TrimPresetsProcessing,
                univ_pro.trim.UNIV_OT_TrimSlotsProcessing,
                univ_pro.trim.UNIV_OT_TrimPresetLoad,
                univ_pro.trim.UNIV_OT_TrimPresetSave,
                univ_pro.trim.UNIV_OT_TrimFromMesh,
                univ_pro.trim.UNIV_OT_TrimEditor,
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
                # Mark
                univ_pro.mark.UNIV_OT_Constraint,
                univ_pro.mark.UNIV_OT_ConstraintByAngle,
                # Misc
                univ_pro.misc.UNIV_OT_TexelDensityFromTexture,
                univ_pro.rectify.UNIV_OT_Rectify,
                univ_pro.projection.UNIV_OT_BoxProject,
                # Unwrap
                univ_pro.unwrap.UNIV_OT_Unwrap,
                # Straight
                univ_pro.straight.UNIV_OT_Straight,

            ))

        else:
            classes.extend((
                # Checker System
                operators.checker.UNIV_OT_Checker,
                # Project
                operators.project.UNIV_OT_BoxProject,
                # Stack
                operators.stack.UNIV_OT_Stack,
                operators.stack.UNIV_OT_Stack_VIEW3D,
                # Unwrap
                operators.unwrap.UNIV_OT_Unwrap,
                # Straight
                operators.straight.UNIV_OT_Straight
            ))
    except AttributeError:
        traceback.print_exc()
        classes = []


def load_workspace_types():
    from . import ui

    classes_workspace.extend([
        ui.UNIV_WT_edit_VIEW3D,
        ui.UNIV_WT_object_VIEW3D]
    )

load_register_types()
load_workspace_types()

is_enabled = False


def univ_load_post_for_timer():
    """For avoid context restrict"""
    # TODO: Add to inspect
    from . import operators
    from . import draw
    draw.shaders.Shaders.init_shaders()
    operators.misc.UNIV_OT_UV_Layers_Manager.uv_layers_watcher_append_handler()

    if univ_pro_exist:
        draw.DrawerSubscribeRNA.register_handler()
        # TODO: Check why this double called ?
        # This was called before without persist (why???)
        draw.DrawerSubscribeRNA.subscribe()  # NOTE: Call after register_handler

        draw.Drawer2D.append_handler_with_delay()
        draw.Drawer3D.append_handler_with_delay()
        draw.TrimDrawer.append_handler_with_delay()

@bpy.app.handlers.persistent
def univ_load_post(_):
    univ_load_post_for_timer()


def register():
    if bpy.app.background:
        print("UniV: Skipping registration in background mode")
        return

    from . import operators
    from . import icons
    from . import keymaps
    from . import preferences

    global is_enabled

    if is_enabled or not classes or not classes_workspace:
        if not classes:
            raise AttributeError('UniV: Failed to load operators, try re-enabling or restarting Blender')

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
            print(f"UniV: Failed to register a class {c.__name__}")
            traceback.print_exc()

    for c in classes_workspace:
        try:
            bpy.utils.register_tool(c)
        except Exception:  # noqa
            print(f"UniV: Failed to register a class {c.__name__}")
            traceback.print_exc()

    # Icons register.
    try:
        icons.icons.register_ws_icons_()
        icons.icons.register_icons_()  # NOTE: Need registered AddonPreferences.
    except:  # noqa
        print('UniV: Icons not loaded.')
        traceback.print_exc()


    # WARNING: When modules are reloaded, classes are overwritten and have no registration.
    # To avoid this, it is necessary to use initially registered classes.
    # Perhaps it does not allow to reload operators in a normal way.
    # bpy.types.WindowManager.univ_settings = bpy.props.PointerProperty(type=classes[3])

    # TODO: Add checks to inspect
    # After restarting the add-on, it doesn't work, but older versions of Blender don't have context-restricted mode,
    # so you can load it without any delay
    # bpy.app.handlers.load_post.append(univ_load_post)
    from bpy.app.timers import register
    register(univ_load_post_for_timer, first_interval = 0.0, persistent = True)

    bpy.types.VIEW3D_HT_header.prepend(operators.toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.prepend(operators.toggle.univ_header_sync_btn)
    bpy.types.IMAGE_HT_header.prepend(operators.toggle.univ_header_split_btn)

    bpy.types.VIEW3D_MT_object_apply.prepend(operators.misc.draw_smart_scale_menu)

    if "NOT_BL_EXT":
        import platform
        if platform.system() in ('Windows', 'Linux'):
            try:
                fastapi.clib.FastAPI.load()
            except:  # noqa
                pass
                # if univ_pro:
                #     print('UniV: Cannot load fastapi.')
                #     traceback.print_exc()

    try:
        keymaps.add_keymaps()
        keymaps.add_keymaps_ws()
    except AttributeError:
        traceback.print_exc()

    preferences.update_panel(None, None)
    operators.toggle.ToggleHandlers.register_handler()


def unregister():
    if bpy.app.background:
        return

    from . import operators
    from . import draw
    from . import icons
    from . import keymaps

    keymaps.remove_keymaps()
    keymaps.remove_keymaps_ws()
    icons.icons.unregister_icons_()
    # icons.icons.unregister_ws_icons_()

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

    bpy.types.VIEW3D_HT_header.remove(operators.toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.remove(operators.toggle.univ_header_split_btn)
    bpy.types.IMAGE_HT_header.remove(operators.toggle.univ_header_sync_btn)
    bpy.types.VIEW3D_MT_object_apply.remove(operators.misc.draw_smart_scale_menu)

    if univ_pro_exist:
        from . import univ_pro
        univ_pro.misc.UNIV_OT_TexelDensityFromTexture.store_poliigon_physical_size_cache()

    operators.toggle.ToggleHandlers.unregister_handler()

    for handler in reversed(bpy.app.handlers.load_post):
        if handler.__name__ == univ_load_post.__name__:
            bpy.app.handlers.load_post.remove(handler)

if __name__ == "__main__":
    register()
