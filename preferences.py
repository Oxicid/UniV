# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import rna_keymap_ui  # type: ignore[import-untyped]

from . import utils
from . import keymaps
from bpy.props import *

UV_LAYERS_ENABLE = True


def prefs() -> 'UNIV_AddonPreferences':
    return bpy.context.preferences.addons[__package__].preferences


def univ_settings() -> 'UNIV_Settings':
    return bpy.context.scene.univ_settings  # noqa


def force_debug():
    return prefs().debug == 'FORCE'


def debug():
    return prefs().debug == 'ENABLED'


def stable():
    return prefs().mode == 'STABLE'


def experimental():
    return prefs().mode == 'EXPERIMENTAL'


def _update_size_x(_self, _context):
    if univ_settings().lock_size and univ_settings().size_y != univ_settings().size_x:
        univ_settings().size_y = univ_settings().size_x


def _update_size_y(_self, _context):
    if univ_settings().lock_size and univ_settings().size_x != univ_settings().size_y:
        univ_settings().size_x = univ_settings().size_y


def _update_lock_size(_self, _context):
    if univ_settings().lock_size and univ_settings().size_y != univ_settings().size_x:
        univ_settings().size_y = univ_settings().size_x


def update_panel(_self, _context):
    from .ui import UNIV_PT_General_VIEW_3D as Panel
    if Panel.bl_category != prefs().panel_3d_view_category:
        try:
            if "bl_rna" in Panel.__dict__:
                bpy.utils.unregister_class(Panel)

            Panel.bl_category = prefs().panel_3d_view_category
            bpy.utils.register_class(Panel)
        except Exception as e:
            print(f'UniV: Updating Panel View 3D category has failed:\n{e}')


def _update_drawer_2d_enable(self, context):
    from . import draw
    draw.Drawer2D.update_data(self, context)

def _update_drawer_3d_enable(self, context):
    from . import draw
    draw.Drawer3D.update_data(self, context)

def _update_overlay_3d_uv_face_color(self, context):
    from . import draw
    draw.shaders.Shaders.init_flat_shading_uniform_color()
    draw.Drawer3D.update_data(self, context)

def _update_uv_layers_show(_self, _context):
    from .operators.misc import UNIV_OT_UV_Layers_Manager
    if _self.uv_layers_show:
        if not any(handler is UNIV_OT_UV_Layers_Manager.univ_uv_layers_update for handler in bpy.app.handlers.depsgraph_update_post):
            bpy.app.handlers.depsgraph_update_post.append(UNIV_OT_UV_Layers_Manager.univ_uv_layers_update)
        from . import ui
        ui.REDRAW_UV_LAYERS = True
    else:
        for handler in reversed(bpy.app.handlers.depsgraph_update_post):
            if handler is UNIV_OT_UV_Layers_Manager.univ_uv_layers_update:
                bpy.app.handlers.depsgraph_update_post.remove(handler)


def _update_uv_layers_name(_self, context):
    if UV_LAYERS_ENABLE:
        settings = univ_settings()
        idx = settings.uv_layers_active_idx
        uv_name = settings.uv_layers_presets[idx].name
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                uvs = obj.data.uv_layers
                if len(obj.data.uv_layers) >= idx+1:
                    if uvs[idx].name != uv_name:
                        uvs[idx].name = uv_name
        from .operators.misc import UNIV_OT_UV_Layers_Manager
        UNIV_OT_UV_Layers_Manager.update_uv_layers_props()


def _update_uv_layers_active_idx(self, context):
    if UV_LAYERS_ENABLE:
        if is_edit := bpy.context.mode == 'EDIT_MESH':
            objects = context.objects_in_mode_unique_data
        else:
            objects = (obj_ for obj_ in context.selected_objects if obj_.type == 'MESH')

        idx = self.uv_layers_active_idx
        for obj in objects:
            uvs = obj.data.uv_layers
            if len(obj.data.uv_layers) >= idx+1:
                if not uvs[idx].active:
                    uvs[idx].active = True
        if prefs().enable_uv_layers_sync_borders_seam and is_edit:
            if area := bpy.context.area:
                if area.type == 'VIEW_3D':
                    bpy.ops.mesh.univ_seam_border(selected=False, mtl=False, by_sharps=False)  # noqa
                else:
                    bpy.ops.uv.univ_seam_border(selected=False, mtl=False, by_sharps=False)  # noqa

        from .operators.misc import UNIV_OT_UV_Layers_Manager
        UNIV_OT_UV_Layers_Manager.update_uv_layers_props()


_udim_source = [
    ('CLOSEST_UDIM', 'Closest UDIM', "Pack islands to closest UDIM"),
    ('ACTIVE_UDIM', 'Active UDIM', "Pack islands to active UDIM image tile or UDIM grid tile where 2D cursor is located")
]
copy_to_layers_uv_channels_items_from = [
    ('0', 'Active', ''),
    ('1', '1', ''),
    ('2', '2', ''),
    ('3', '3', ''),
    ('4', '4', ''),
    ('5', '5', ''),
    ('6', '6', ''),
    ('7', '7', ''),
    ('8', '8', ''),
]
copy_to_layers_uv_channels_items_to = copy_to_layers_uv_channels_items_from.copy()
copy_to_layers_uv_channels_items_to[0] = ('0', 'Other', '')

if _is_360_pack := bpy.app.version >= (3, 6, 0):
    _udim_source.append(('ORIGINAL_AABB', 'Original BBox', "Pack to starting bounding box of islands"))


class UNIV_TexelPreset(bpy.types.PropertyGroup):
    texel: FloatProperty(name='Texel', default=512, min=1, max=20_000)
    size_x: EnumProperty(name='Size X', default='2048', items=utils.resolutions)
    size_y: EnumProperty(name='Size Y', default='2048', items=utils.resolutions)


class UNIV_UV_Layers(bpy.types.PropertyGroup):
    name: StringProperty(name='UVMap', update=_update_uv_layers_name)
    flag: IntProperty(name='Flag', default=0, min=0, max=3)


class UNIV_Settings(bpy.types.PropertyGroup):
    # Global Settings
    size_x: EnumProperty(name='X', default='2048', items=utils.resolutions, update=_update_size_x)
    size_y: EnumProperty(name='Y', default='2048', items=utils.resolutions, update=_update_size_y)
    lock_size: BoolProperty(name='Lock Size', default=True, update=_update_lock_size)

    # Texel Settings
    texel_density: FloatProperty(name="Texel Density", default=512, min=1, max=10_000, precision=1,
                                 description="The number of texture pixels (texels) per unit surface area in 3D space.")
    active_td_index: IntProperty(min=0, max=8, options={'SKIP_SAVE'})
    texels_presets: CollectionProperty(name="TD Presets", type=UNIV_TexelPreset)
    texture_physical_size: FloatVectorProperty(name='TD from Physical Size', default=(2.5, 0.0), min=0.0,
                                                    soft_max=6, size=2, subtype='TRANSLATION')

    # Drawer
    overlay_2d_enable: BoolProperty(name='Overlay 2D', default=True, update=_update_drawer_2d_enable)
    overlay_3d_enable: BoolProperty(name='Overlay 3D', default=True, update=_update_drawer_3d_enable)

    overlay_3d_uv_vert_color: FloatVectorProperty(name="UV Select Vert", default=(1.0, 1.0, 1.0, 0.28),
        min=0.0, max=1.0, size=4, subtype='COLOR'
    )
    overlay_3d_uv_edge_color: FloatVectorProperty(name="UV Select Edge", default=(0.82, 0.82, 0.82, 0.72),
        min=0.0, max=1.0, size=4, subtype='COLOR'
    )
    overlay_3d_uv_face_color: FloatVectorProperty(name="UV Select Face", default=(0.0, 0.196, 0.7, 0.45),
        min=0.0, max=1.0, size=4, subtype='COLOR', update=_update_overlay_3d_uv_face_color
    )
    overlay_toggle_xray: BoolProperty(name='Toggle X-Ray', default=True,
        description="Shows faces through geometry. Vertices and edges are always visible in this mode, otherwise they are not visible.")

    # UV Layer
    uv_layers_show: BoolProperty(name='Show UV Layers', default=True, update=_update_uv_layers_show)

    uv_layers_size: IntProperty(name='Size', min=0, max=8, default=0, options={'SKIP_SAVE'})
    uv_layers_active_idx: IntProperty(name='Active UV index', min=0, max=7, default=0,
                                      update=_update_uv_layers_active_idx, options={'SKIP_SAVE'})
    uv_layers_active_render_idx: IntProperty(name='Active uv render index',
                                             min=-1, max=7, default=-1, options={'SKIP_SAVE'})
    uv_layers_presets: CollectionProperty(name="UV Layers", type=UNIV_UV_Layers, options={'SKIP_SAVE'})

    copy_to_layers_from: EnumProperty(name='From', default='0', items=copy_to_layers_uv_channels_items_from)
    copy_to_layers_to: EnumProperty(name='To', default='0', items=copy_to_layers_uv_channels_items_to)

    # Pack Settings
    use_uvpm: BoolProperty(name='Use UVPackmaster', default=False)
    shape_method: EnumProperty(name='Shape Method', default='CONCAVE',
                               items=(('CONCAVE', 'Exact', 'Uses exact geometry'),
                                      ('AABB', 'Fast', 'Uses bounding boxes'))
                               )
    scale: BoolProperty(name='Scale', default=True, description="Scale islands to fill unit square")
    rotate: BoolProperty(name='Rotate', default=True, description="Rotate islands to improve layout")
    rotate_method: EnumProperty(name='Rotation Method', default='CARDINAL',
                                items=(
                                    ('ANY', 'Any', "Any angle is allowed for rotation"),
                                    ('AXIS_ALIGNED', 'Orient', "Rotated to a minimal rectangle, either vertical or horizontal"),
                                    ('CARDINAL', 'Step 90', "Only 90 degree rotations are allowed")

                                ))

    pin: BoolProperty(name='Lock Pinned Islands', default=False,
                      description="Constrain islands containing any pinned UV's")
    pin_method: EnumProperty(name='Lock Method', default='LOCKED',
                             items=(
                                 ('LOCKED', 'All', "Pinned islands are locked in place"),
                                 ('ROTATION_SCALE', 'Rotation and Scale', "Pinned islands will translate only"),
                                 ('ROTATION', 'Rotation', "Pinned islands won't rotate"),
                                 ('SCALE', 'Scale', "Pinned islands won't rescale")))

    merge_overlap: BoolProperty(name='Lock Overlaps', default=False)
    udim_source: EnumProperty(name='Pack to', default='CLOSEST_UDIM', items=_udim_source)

    padding: IntProperty(name='Padding', default=8, min=0, soft_min=2, soft_max=32, max=64, step=2,
                         subtype='PIXEL', description="Space between islands in pixels.\n\n"
                                                      "Formula for converting the current Padding implementation to Margin:\n"
                                                      "Margin = Padding / 2 / Texture Size\n\n"
                                                      "Optimal value for UV padding:\n"
                                                      "256 = 1  px\n"
                                                      "512 = 2-3 px\n"
                                                      "1024 = 4-5 px\n"
                                                      "2048 = 8-10 px\n"
                                                      "4096 = 16-20 px\n"
                                                      "8192 = 32-40 px\t")

    align_mode: EnumProperty(name="Align Mode", default='ALIGN', items=(
        ('ALIGN', 'Align', 'Align', 'EMPTY_SINGLE_ARROW', 0),
        ('INDIVIDUAL_OR_MOVE', 'Individual | Move', 'Individual Align. Move in Island Mode. '
                                                    'Collect in Island mode when press Center. '
                                                    'HV applies align by edge angle in Island mode', 'PIVOT_INDIVIDUAL', 1),
        ('ALIGN_TO_CURSOR', 'Align to cursor', 'Align to cursor', 'PIVOT_CURSOR', 2),
        ('ALIGN_TO_CURSOR_UNION', 'Align to cursor union', 'Align to cursor union', 'EVENT_U', 3)
    ))

    align_island_mode: EnumProperty(name="Island Mode", default='FOLLOW', items=(
        ('FOLLOW', 'Follow', '', 'EVENT_F', 0),
        ('ISLAND', 'Island', '', 'UV_ISLANDSEL', 1),
        ('VERTEX', 'Vertex', '', 'VERTEXSEL', 2)
    ))

    batch_inspect_flags: IntProperty(name="Batch Inspect Tags", min=0,
                                     default=__import__(__package__.replace('preferences', '') + '.operators.inspect',
                                                        fromlist=['inspect']).Inspect.default_value_for_settings()
                                     )


class UNIV_AddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    tab: EnumProperty(
        items=(
            ('GENERAL', 'General', ''),
            ('UI', 'UI', ''),
            ('KEYMAPS', 'Keymaps', ''),
            ('INFO', 'Info', ''),
        ),
        default='KEYMAPS')
    # default='INFO')  # noqa

    debug: EnumProperty(name='Debug',
                        items=(
                            ('DISABLED', 'Disabled', ''),
                            ('ENABLED', 'Enabled', ''),
                            ('FORCE', 'Force', ''),
                        ),
                        default='DISABLED')

    mode: EnumProperty(name='Mode',
                       items=(
                           ('STABLE', 'Stable', ''),
                           ('EXTENDED', 'Extended', ''),
                           ('EXPERIMENTAL', 'Experimental', ''),
                       ),
                       default='EXTENDED')

    use_csa_mods: bpy.props.BoolProperty(default=True,
                                         name="Use Modifier Keys",
                                         description="Enable behavior changes based on Ctrl, Shift, and Alt keys when invoking the operator"
                                         )

    snap_points_default: EnumProperty(name='Default Snap Points',
                                      items=(
                                          ('ALL', 'All', ''),
                                          ('FOLLOW_MODE', 'Follow Mode', 'Follow the selection mode, VERTEX mode remains always')
                                      ),
                                      default='FOLLOW_MODE',
                                      description='Default Snap Points for QuickSnap')

    # ----------
    color_mode: EnumProperty(name='Color Mode',
                             items=(('COLOR', 'Color', ''), ('MONO', 'Monochrome', '')),
                             default='COLOR',
                             update=lambda self, _: __import__(__package__.replace('preferences', '')+'.icons', fromlist=['icons']).icons.register_icons_())
    icon_size: EnumProperty(name='Icon Size',
                            items=(('32', '32', ''), ('64', '64', ''), ('128', '128', ''), ('256', '256', '')),
                            default='32')
    icon_antialiasing: EnumProperty(name='Anti-Aliasing',
                                    items=(('1', 'x1', ''), ('2', 'x2', ''), ('4', 'x4', ''), ('8', 'x8', '')),
                                    default='4')
    # Mono
    icon_mono_green: FloatVectorProperty(name='Green',
                                         # 8bc6a1
                                         subtype="COLOR", size=4, min=0.0, max=1.0, default=(0.258181, 0.564712, 0.3564003, 1))

    icon_mono_gray: FloatVectorProperty(name='Grey Color',
                                        # c7c7c7
                                        subtype="COLOR", size=4, min=0.0, max=1.0, default=(0.5711, 0.5711, 0.5711, 1))

    # Colored
    icon_colored_violet: FloatVectorProperty(name='Violet',
                                             # 7d87ff
                                             subtype="COLOR", size=4, min=0.0, max=1.0, default=(0.20505, 0.24228, 1, 1))

    icon_colored_cian: FloatVectorProperty(name='Cian',
                                           # 62cdf9
                                           subtype="COLOR", size=4, min=0.0, max=1.0, default=(0.1221, 0.6105, 0.9473, 1))

    icon_colored_purple: FloatVectorProperty(name='Purple',
                                             # dc87ff
                                             subtype="COLOR", size=4, min=0.0, max=1.0, default=(0.71569, 0.24228, 1, 1))

    icon_colored_pink: FloatVectorProperty(name='Pink',
                                           # ff87a9
                                           subtype="COLOR", size=4, min=0.0, max=1.0, default=(1, 0.24228, 0.396755, 1))
    # Common
    icon_white_color: FloatVectorProperty(name='White Color',
                                          subtype="COLOR", size=4, min=0.0, max=1.0, default=(1, 1, 1, 1))  # ffffff

    icon_select_arrow_color: FloatVectorProperty(name='Select Arrow Color',
                                                 # ececec
                                                 subtype="COLOR", size=4, min=0.0, max=1.0, default=(0.83879, 0.83879, 0.83879, 1))

    # ----------
    keymap_workspace_filter: EnumProperty(name='Workspace Filter',
                                          items=(
                                              ('ALL', 'Show All', ''),
                                              ('DEFAULT', 'Default', 'Show keymaps from default workspaces'),
                                              ('WORKSPACE', 'Workspace', 'Show keymaps from univ workspaces')
                                          ),
                                          default='ALL')

    keymap_spaces_filter: EnumProperty(name='Space Filter',
                                       items=(
                                           ('ALL', 'Show All Spaces', ''),
                                           ('UV Editor', 'UV', ''),
                                           ('Window', 'Window', ''),
                                           ('Object', 'Object', ''),
                                           ('Mesh', 'Mesh', ''),

                                       ),
                                       default='ALL')

    keymap_conflict_filter: EnumProperty(name='Conflict Filter',
                                         items=(
                                             ('ALL', 'Show All', ''),
                                             ('ONLY_ERROR', 'Only Error', ''),
                                             ('HIDE_ERROR', 'Hide Error', ''),

                                         ),
                                         default='ALL')

    keymap_name_filter: StringProperty(name="Search by Name", default='', options={'TEXTEDIT_UPDATE'})
    keymap_key_filter: StringProperty(name="Search by Key-Binding", default='', options={'TEXTEDIT_UPDATE'})

    split_toggle_uv_by_cursor: BoolProperty(name='Split ToggleUV by Mouse Cursor', default=False)
    show_split_toggle_uv_button: BoolProperty(name='Show Split ToggleUV Button', default=False)
    show_view_3d_panel: BoolProperty(name='Show View 3D Panel', default=True)
    panel_3d_view_category: StringProperty(name="Panel 3D View Category", description="Enter a name for the panel category",
                                           default='UniV', update=update_panel)
    # enable_uv_name_controller: BoolProperty(name='Enable UV name controller', default=False)
    enable_uv_layers_sync_borders_seam: BoolProperty(name='Enable sync Border Seam', default=True)

    max_pick_distance: IntProperty(name='Max Pick Distance', default=75, min=15, soft_max=100, subtype='PIXEL',
                                   description='Pick Distance for Pick Select, Quick Snap operators'
                                   )

    def draw(self, _context):
        layout = self.layout
        row = layout.row()
        row.prop(self, "tab", expand=True)

        if self.tab == 'GENERAL':
            layout.prop(self, 'debug')
            layout.prop(self, 'mode')
            layout.prop(self, 'use_csa_mods')
            layout.separator()

            layout.label(text='QuickSnap:')
            layout.prop(self, 'snap_points_default')
            layout.separator()

            layout.prop(self, 'max_pick_distance')
            layout.prop(self, 'enable_uv_layers_sync_borders_seam')

        elif self.tab == 'UI':
            layout.prop(self, 'show_view_3d_panel')
            layout.prop(self, 'panel_3d_view_category')
            row = layout.row()
            row.prop(self, 'split_toggle_uv_by_cursor')
            row.prop(self, 'show_split_toggle_uv_button')
            layout.separator()
            box = layout.box()
            box.prop(self, 'color_mode')

            split = box.split()
            col = split.column()

            if self.color_mode == 'MONO':
                col.label(text="Green Color:")
                col.label(text="Gray Color:")
            else:
                col.label(text="Violet Color:")
                col.label(text="Cian Color:")
                col.label(text="Purple Color:")
                col.label(text="Pink Color:")

            col.label(text="White Color:")
            col.label(text="Select Arrow Color:")
            col = split.column()
            if self.color_mode == 'MONO':
                col.prop(self, "icon_mono_green", text="")
                col.prop(self, "icon_mono_gray", text="")
            else:
                col.prop(self, "icon_colored_violet", text="")
                col.prop(self, "icon_colored_cian", text="")
                col.prop(self, "icon_colored_purple", text="")
                col.prop(self, "icon_colored_pink", text="")

            col.prop(self, "icon_white_color", text="")
            col.prop(self, "icon_select_arrow_color", text="")

            box_split = box.row()
            box_split.prop(self, 'icon_size')
            box_split.prop(self, 'icon_antialiasing', text='AA')
            box_split = box.row(align=True)
            box_split.operator('wm.univ_icons_generator', text='Generate icons').generate_only_ws_tool_icon = False
            box_split.operator('wm.univ_icons_generator',
                               text='Generate only Workspace Tool icons').generate_only_ws_tool_icon = True

        elif self.tab == 'KEYMAPS':
            row = layout.row()
            row.operator('wm.univ_keymaps_config', text='Restore').mode = 'RESTORE'
            row.operator('wm.univ_keymaps_config', text='Off/On').mode = 'TOGGLE'
            row.operator('wm.univ_keymaps_config', text='Delete User').mode = 'DELETE_USER'
            row.operator('wm.univ_keymaps_config', text='Resolve Conflicts').mode = 'RESOLVE_ALL'

            col = layout.column(align=True)
            row = col.row(align=True)
            sub_row = row.row(align=True)
            sub_row.scale_x = 0.4
            sub_row.prop(self, 'keymap_workspace_filter', expand=True)

            row.separator()
            sub_row = row.row()
            sub_row.prop(self, 'keymap_spaces_filter', text='')

            row = col.row(align=True)
            sub_row = row.row()
            sub_row.scale_x = 0.8
            sub_row.prop(self, 'keymap_conflict_filter', expand=True)
            row.separator()

            sub_row = row.row(align=True)
            if bpy.app.version >= (4, 0, 0):
                sub_row.prop(self, "keymap_name_filter", text="", icon='SORTALPHA', placeholder="Search by Name")
                sub_row.prop(self, "keymap_key_filter", text="", icon='KEYINGSET', placeholder="Search by Key-Binding")
            else:
                sub_row.prop(self, "keymap_name_filter", text="", icon='SORTALPHA')
                sub_row.prop(self, "keymap_key_filter", text="", icon='KEYINGSET')

            # Draw Keymaps
            if self.keymap_workspace_filter in ('ALL', 'DEFAULT'):
                if self.keymap_spaces_filter == 'ALL':
                    areas = keymaps.keys_areas
                else:
                    areas = []
                    for a in keymaps.keys_areas:
                        if self.keymap_spaces_filter in a:
                            areas.append(a)
                            break
                it = keymaps.ConflictFilter.get_conflict_filtered_keymaps_with_exclude(areas)
                for area, kc, km, filtered_keymaps in it:
                    layout.label(text=area)
                    col = layout.column(align=True)

                    for config_filtered in filtered_keymaps.values():
                        box = None

                        for univ_kmi in config_filtered.univ_keys:
                            any_active = any(univ_kmi.active for univ_kmi in config_filtered.univ_keys)

                            conflict_state_default_keys = 'NONE'
                            if config_filtered.default_keys:
                                if any_active and any(kmi_.active for (_, kmi_) in config_filtered.default_keys):
                                    conflict_state_default_keys = 'ERROR'

                            conflict_state_user_defined = 'NONE'
                            if config_filtered.default_keys:
                                if any_active and any(kmi_.active for (_, kmi_) in config_filtered.user_defined):
                                    conflict_state_user_defined = 'ERROR'

                            if self.keymap_conflict_filter == 'ONLY_ERROR':
                                if 'ERROR' not in (conflict_state_default_keys, conflict_state_user_defined):
                                    continue

                            if not box:
                                box = col.box()

                            rna_keymap_ui.draw_kmi([], kc, km, univ_kmi, box, 0)
                            if self.keymap_conflict_filter != 'HIDE_ERROR':
                                self.draw_conflict_keymaps(box, config_filtered, kc)

            if self.keymap_workspace_filter in ('ALL', 'WORKSPACE'):
                layout.label(text='Workspace Tool')
                if self.keymap_spaces_filter == 'ALL':
                    areas = keymaps.keys_areas_workspace
                else:
                    areas = []
                    for a in keymaps.keys_areas_workspace:
                        if self.keymap_spaces_filter in a:
                            areas.append(a)
                            break

                it = keymaps.ConflictFilter.get_conflict_filtered_keymaps_with_exclude_ws(areas)
                for area, kc, km, filtered_keymaps in it:
                    layout.label(text=area)
                    col = layout.column(align=True)

                    for config_filtered in filtered_keymaps.values():
                        box = None

                        for univ_kmi in config_filtered.univ_keys:
                            any_active = any(univ_kmi.active for univ_kmi in config_filtered.univ_keys)

                            conflict_state_default_keys = 'NONE'
                            if config_filtered.default_keys:
                                if any_active and any(kmi_.active for (_, kmi_) in config_filtered.default_keys):
                                    conflict_state_default_keys = 'ERROR'

                            conflict_state_user_defined = 'NONE'
                            if config_filtered.default_keys:
                                if any_active and any(kmi_.active for (_, kmi_) in config_filtered.user_defined):
                                    conflict_state_user_defined = 'ERROR'

                            if self.keymap_conflict_filter == 'ONLY_ERROR':
                                if 'ERROR' not in (conflict_state_default_keys, conflict_state_user_defined):
                                    continue
                                if univ_kmi.idname == 'view3d.select_box':
                                    continue

                            if not box:
                                box = col.box()

                            rna_keymap_ui.draw_kmi([], kc, km, univ_kmi, box, 0)
                            if univ_kmi.idname == 'view3d.select_box':
                                continue

                            if self.keymap_conflict_filter != 'HIDE_ERROR':
                                self.draw_conflict_keymaps(box, config_filtered, kc)

            layout.label(
                text='To restore deleted keymaps, just reload the addon. But it is better to use the checkboxes to disable them',
                icon='INFO')

        # elif self.tab == 'INFO':
        else:
            enable = True
            if hasattr(bpy.app, 'online_access'):
                enable = bpy.app.online_access
            row = layout.row(align=True)
            row.enabled = enable
            row.operator("wm.url_open", text="YouTube").url = r"https://www.youtube.com/@oxicid6058"
            row.operator("wm.url_open", text="Discord").url = r"https://discord.gg/SAvEbGTkjR"
            row.operator("wm.url_open", text="GitHub").url = r"https://github.com/Oxicid/UniV"
            row.operator("wm.url_open", text="Blender Market").url = r"https://blendermarket.com/products/univ?search_id=32308413"

            from .icons import icons

            try:
                from . import univ_pro
            except ImportError:
                univ_pro = None

            if not univ_pro:
                layout.label(
                    text="You have the free version of the addon installed, which does not have some advanced operators and options", icon='INFO')
                layout.label(text="which does not have some advanced operators and options")
            else:
                layout.label(text="You are using the Pro version", icon='FAKE_USER_ON')

            layout.label(text="UniV Pro includes such advanced operators as:")
            layout.label(text="Rectify - straightens the island by selected 4 boundary vertices, works also with triangles and N-Gone too.",
                         icon_value=icons.rectify)
            layout.separator(factor=0.35)
            layout.label(text="Transfer - interactively transfers a UV layer from one object to another.",
                         icon_value=icons.transfer)
            layout.separator(factor=0.35)
            layout.label(text="Select by Flat [2D and 3D] - select linked flat faces by angle", icon_value=icons.flat)
            layout.separator(factor=0.35)
            layout.label(text="Loop Select [2D and 3D] [Ctrl+Alt+WheelUp] - edge loop select, works also with triangles and N-Gone too.",
                         icon_value=icons.loop_select)
            layout.separator(factor=0.35)
            layout.label(text="Drag - this operator is similar to the QuickSnap operator, but has fundamental differences:",
                         icon_value=icons.fill)
            layout.label(text="     1) It works only with islands")
            layout.label(text="     2) Moves only one island")
            layout.label(text="     3) Unselects all other elements and selects the picked island.")
            layout.label(
                text="     4) Faster manipulation, LMB + Alt + Drag moves the islands, if you release LMB - the operator ends.")
            layout.label(text="     5) Can pull out overlapped flipped islands.")
            layout.label(text="     6) Snapping is not a key and intrusive feature.")
            layout.separator(factor=0.35)
            layout.label(text="Stack - has more advanced options such as working with symmetrical UV islands as well as working with Mesh islands ",
                         icon_value=icons.stack)
            layout.label(
                text="Select Similar - Selects similar islands, useful in combination with the Stack operator ", icon_value=icons.arrow)
            layout.label(
                text="Box Projection Pro- Has an Orient 3D option, which tightly projects cube-shaped elements ", icon_value=icons.box)
            if not univ_pro:
                layout.label(text="You can get the Pro version for free in the Discord channel.", icon='INFO')

    @staticmethod
    def draw_conflict_keymaps(box, config_filtered, kc):
        if config_filtered.default_keys:
            for (default_km, default_kmi) in config_filtered.default_keys:
                box_split = box.split(align=True, factor=0.5)
                box_split.label(text=' ', icon='ERROR' if default_kmi.active else 'BLANK1')
                rna_keymap_ui.draw_kmi([], kc, default_km, default_kmi, box_split, 0)
        if config_filtered.user_defined:
            for (user_km, user_kmi) in config_filtered.user_defined:
                box_split = box.split(align=True, factor=0.5)
                box_split.label(text=' ', icon='ERROR' if user_kmi.active else 'BLANK1')
                rna_keymap_ui.draw_kmi([], kc, user_km, user_kmi, box_split, 0)


class UNIV_OT_ShowAddonPreferences(bpy.types.Operator):
    bl_idname = 'wm.univ_show_addon_preferences'
    bl_label = 'Addon Preferences'

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        context.preferences.active_section = 'ADDONS'
        context.window_manager.addon_search = 'UniV'
        return {'FINISHED'}
