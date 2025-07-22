# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

# The icons were created by Vitaly Zhdanov https://www.youtube.com/@diffusecolor , for which he is very thankful!
# His work gave the project a finished and professional look.
# Excellent detailing and stylish design made the interface more convenient and pleasant.
# Thank you for the work done!

import os
import gpu
import bpy
import numpy as np
from mathutils import Vector, Matrix, Color
from pathlib import Path

class icons:
    """
    Interface for accessing icons.

    NOTE: Icon names must match the names of the corresponding PNG files.
    NOTE: For properties, methods, and other attributes that don’t contain an icon value, names must start with an underscore.
    """
    _icons_ = None
    adjust = 0
    area = 0
    arrow = 0
    arrow_bottom = 0
    arrow_bottom_left = 0
    arrow_bottom_right = 0
    arrow_left = 0
    arrow_right = 0
    arrow_top = 0
    arrow_top_left = 0
    arrow_top_right = 0
    border = 0
    border_by_angle = 0
    border_seam = 0
    box = 0
    center = 0
    checker = 0
    coverage = 0
    crop = 0
    cursor = 0
    cut = 0
    distribute = 0
    edge_grow = 0
    fill = 0
    flat = 0
    flip = 0
    flipped = 0
    gravity = 0
    grow = 0
    home = 0
    horizontal_a = 0
    horizontal_c = 0
    large = 0
    linked = 0
    loop_select = 0
    medium = 0
    non_splitted = 0
    normal = 0
    normalize = 0
    orient = 0
    over = 0
    overlap = 0
    pack = 0
    pin = 0
    quadrify = 0
    random = 0
    rectify = 0
    relax = 0
    remove = 0
    reset = 0
    rotate = 0
    select_stacked = 0
    settings_a = 0
    settings_b = 0
    shift = 0
    small = 0
    smart = 0
    sort = 0
    square = 0
    stack = 0
    stitch = 0
    straight = 0
    td_get = 0
    td_set = 0
    transfer = 0
    unwrap = 0
    vertical_a = 0
    vertical_b = 0
    view = 0
    weld = 0
    x = 0
    y = 0
    zero = 0

    @classmethod
    def register_icons_(cls):
        from bpy.utils import previews
        if cls._icons_:
            cls.unregister_icons_()
        cls._icons_ = previews.new()

        from ..preferences import prefs
        png_folder_name = 'png_mono/' if prefs().color_mode == 'MONO' else 'png/'
        png_file_path = __file__.replace('__init__.py', png_folder_name)

        for attr in dir(cls):
            if not attr.endswith('_'):
                if not isinstance(getattr(cls, attr), int):
                    print(f'UniV: Attribute {attr} not icon_id')
                    continue

                full_path = png_file_path + attr + ".png"
                if not os.path.exists(full_path):
                    print(f'UniV: File {full_path} for {attr} icon not found')
                    continue

                icon = cls._icons_.load(attr, full_path, 'IMAGE')
                _ = icon.icon_pixels[0]  # Need to force load icons, bugreport?
                setattr(cls, attr, icon.icon_id)

        # Updating icons for workspaces
        from pathlib import Path
        from .. import ui

        panels = [ui.UNIV_WT_edit_VIEW3D, ui.UNIV_WT_object_VIEW3D]

        icon_path = Path(panels[0].bl_icon)
        expected_icon = 'univ_mono' if prefs().color_mode == 'MONO' else 'univ'

        if icon_path.parts[-1] != expected_icon:
            from .. import keymaps
            keymaps.remove_keymaps_ws()
            new_path = icon_path.parent / expected_icon
            for p in panels:
                try:
                    bpy.utils.unregister_tool(p)
                    p.bl_icon = str(new_path)
                    bpy.utils.register_tool(p)
                except Exception as e:
                    print(f'UniV: Updating icons for workspaces has failed:\n{e}')
            keymaps.add_keymaps_ws()

    @classmethod
    def reset_icon_value_(cls):
        for attr in dir(cls):
            if not attr.endswith('_'):
                setattr(cls, attr, 0)

    @classmethod
    def unregister_icons_(cls):
        from bpy.utils import previews
        try:
            previews.remove(cls._icons_)
        except KeyError:
            from ..preferences import debug
            if debug():
                import traceback
                traceback.print_exc()

        cls.reset_icon_value_()

class PreviousData:
    """Save and restore all data, since color changes require importing SVG objects."""
    def __init__(self):
        self.prev_mode = bpy.context.mode
        self.prev_active_obj = bpy.context.view_layer.objects.active
        self.prev_sel_object = list(bpy.context.selected_objects)
        self.prev_objects = set(bpy.context.scene.objects)
        self.prev_collections = set(bpy.data.collections)
        self.prev_materials = set(bpy.data.materials)
        self.prev_meshes = set(bpy.data.meshes)
        self.prev_curves = set(bpy.data.curves)

    def restore(self):
        for obj in set(bpy.context.scene.objects) - self.prev_objects:
            bpy.data.objects.remove(obj)
        for col in set(bpy.data.collections) - self.prev_collections:
            bpy.data.collections.remove(col)
        for mat in set(bpy.data.materials) - self.prev_materials:
            bpy.data.materials.remove(mat)
        for mesh in set(bpy.data.meshes) - self.prev_meshes:
            bpy.data.meshes.remove(mesh)
        for curves in set(bpy.data.curves) - self.prev_curves:
            bpy.data.curves.remove(curves)

        for obj in self.prev_sel_object:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = self.prev_active_obj
        if bpy.ops.object.mode_set.poll():
            if self.prev_mode == 'EDIT_MESH':
                bpy.ops.object.mode_set(mode='EDIT')
            else:
                bpy.ops.object.mode_set(mode=self.prev_mode)

class WSToolIconsGenerator:
    """
    Generates .dat format.

    1) To generate an icon for WST, follow these steps:
    2) Model the icon as a polygonal mesh.
    3) The icon mesh must be centered at the origin and fit within the range [-1, -1] ... [1, 1].
    4) Separate different colors into separate objects.
    5) For each object, run WSToolIconsGenerator.pprint to print its data to the console, then copy the printed output from the console.
    6) Create a dedicated method for each part (object) using the printed data — see get_u_tris as an example.
    7) Insert the generated methods into create_batch_from_tris_and_colors, and assign per-vertex colors for each triangle as shown in that method.
    8) Adapt WSToolIconsGenerator.update_wst_icon to your specific WST classes.
    9) Run WSToolIconsGenerator.create_dat_icons to generate the icon.

    This will produce the icon, but you’ll also need to replicate and adapt
    the operator UNIV_OT_IconsGenerator and the related properties in AddonPreferences,
    such as color_mode and others tied to icon handling.
    """
    @classmethod
    def pprint(cls):
        """Extracts the triangle coordinates and indices.
        The coordinates are remapped to a more optimized version -1..1 -> 0..255 (float16 to uint8)."""
        obj = bpy.context.object
        mesh = obj.data

        num_verts = len(mesh.vertices)
        verts_np = np.empty(num_verts * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts_np)

        # print flat vertices
        verts_2d = cls.convert_3d_to_2d(verts_np)
        res = cls.remap_f16_to_uint8(verts_2d)

        pretty = 'coords = np.array(['
        for v in res.reshape(-1):
            pretty += str(v)
            pretty += ', '

        pretty = pretty[:-2] + '], dtype=np.uint8).reshape(-1, 2)\n'
        print(pretty)

        # print flat indexes
        mesh.calc_loop_triangles()
        indexes_np = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get("vertices", indexes_np)

        pretty = 'indexes =  np.array(['
        for v in indexes_np:
            pretty += str(v)
            pretty += ', '

        pretty = pretty[:-2] + '], dtype=np.uint8).reshape(-1, 3)'
        print(pretty)

    @staticmethod
    def remap_f16_to_uint8(arr):
        # -1..1 -> 0..255
        arr += 1.0
        arr *= 0.5 * 255
        np.clip(arr, 0, 255, out=arr)
        return np.array(arr, dtype=np.uint8)

    @staticmethod
    def remap_uint8_to_f16(arr):
        # 0..255 -> -1..1
        f_arr = np.divide(arr, (255 / 2), dtype=np.float16)
        f_arr -= 1
        return f_arr

    @staticmethod
    def convert_3d_to_2d(arr):
        return arr.reshape(-1, 3)[:, :2].reshape(-1, 2)

    @classmethod
    def get_u_tris(cls):
        """Returns an array of triangles - [[(10, 20), (40, 80), (60, 20)], ...]"""
        coords = np.array([111, 159, 95, 159, 95, 107, 96, 101, 99, 95, 105, 92, 111, 91, 147, 91, 153, 92, 159,
                           95, 162, 101, 163, 107, 163, 159, 147, 159, 147, 107, 111, 107], dtype=np.uint8).reshape(-1, 2)

        indexes =  np.array([2, 0, 1, 14, 12, 13, 2, 15, 0, 14, 11, 12, 3, 15, 2, 14, 10, 11, 15, 3, 4, 15, 4, 5,
                             15, 5, 6, 9, 14, 8, 9, 10, 14, 8, 14, 7, 14, 6, 7, 14, 15, 6], dtype=np.uint8).reshape(-1, 3)
        return coords[indexes]

    @classmethod
    def get_leaf_tris(cls):
        """Return r up, r bot, l bot, l up tris
        NOTE: This code is specific to UniV - use a simpler method like in get_u_tris.
        """
        # Right upper leaf
        coords = np.array(
            [143, 198, 149, 200, 155, 204, 160, 209, 176, 225, 225, 176, 209, 160, 204, 155, 201, 149, 198, 143, 183,
             143, 185, 150, 189, 157, 193, 164, 199, 170, 204, 176, 176, 204, 171, 198, 164, 193, 158, 188, 151, 185,
             143, 183], dtype=np.uint8).reshape(-1, 2)

        indexes = np.array(
            [15, 4, 16, 16, 3, 17, 17, 2, 18, 1, 19, 2, 9, 8, 10, 0, 21, 1, 12, 8, 7, 7, 14, 13, 18, 2, 19, 7, 13, 12,
             6, 15, 14, 15, 5, 4, 16, 4, 3, 17, 3, 2, 1, 20, 19, 21, 20, 1, 12, 11, 8, 7, 6, 14, 6, 5, 15, 8, 11, 10],
            dtype=np.uint8).reshape(-1, 3)

        flipped_indexes = indexes[:, ::-1]

        r_up_leaf_f16 = cls.remap_uint8_to_f16(coords)
        # Generate other UniV icon leafs coords
        r_bottom_leaf_f16 =  r_up_leaf_f16 * np.array([1, -1], dtype=np.float16)
        l_bottom_leaf_f16 =  r_up_leaf_f16 * np.array([-1, -1], dtype=np.float16)
        l_up_leaf_f16 =  r_up_leaf_f16 * np.array([-1, 1], dtype=np.float16)

        r_u_tris_u8 = coords[indexes]
        # Get other UniV icon leafs coords from indices
        r_b_tris_u8 = cls.remap_f16_to_uint8(r_bottom_leaf_f16)[flipped_indexes]
        l_b_tris_u8 = cls.remap_f16_to_uint8(l_bottom_leaf_f16)[indexes]
        l_u_tris_u8 = cls.remap_f16_to_uint8(l_up_leaf_f16)[flipped_indexes]
        return r_u_tris_u8, r_b_tris_u8, l_b_tris_u8, l_u_tris_u8

    @classmethod
    def create_batch_from_tris_and_colors(cls):
        """
        Creates flat triangle coordinate data, and creates a different color for each vertex of the triangle.

        To make this function easier to understand, remove the code
        related to leafs_colors and leafs_shape, leave only u_shape and u_co
        """
        from ..preferences import prefs
        def convert_float_to_srgb_int(col):
            alpha = col[3]
            col = Color(col[:3])
            col.s *= 1.15
            col.v *= 1.15
            color_srgb = col.from_scene_linear_to_srgb()
            return tuple(round(c * 255) for c in (*color_srgb, alpha))

        if prefs().color_mode == 'MONO':
            u_col = [convert_float_to_srgb_int(prefs().icon_mono_gray)]
            leaf_col = [convert_float_to_srgb_int(prefs().icon_mono_green) for _ in range(4)]
        else:
            u_col = [int(v*255) for v in prefs().icon_white_color]
            leaf_col = [convert_float_to_srgb_int(prefs().icon_colored_pink),
                         convert_float_to_srgb_int(prefs().icon_colored_purple),
                         convert_float_to_srgb_int(prefs().icon_colored_violet),
                         convert_float_to_srgb_int(prefs().icon_colored_cian),
                        ]

        u_shape = cls.get_u_tris()
        # Create a color for each vertex (duplicate one color)
        u_colors_size = len(u_shape)*3
        u_colors = np.array([u_col], dtype=np.uint8).repeat(u_colors_size, axis=0)

        leafs_shape = cls.get_leaf_tris()
        leafs_colors_size = len(leafs_shape[0])*3
        leafs_colors = np.array(leaf_col, dtype=np.uint8).repeat(leafs_colors_size, axis=0)

        # Convert to flat data (first write triangles, then colors)
        shapes = [u_shape, *leafs_shape] + [u_colors, leafs_colors]
        for idx, shape in enumerate(shapes):
            shapes[idx] = shape.reshape(-1)

        union_flat_data = np.concatenate(shapes)
        return union_flat_data.tobytes()

    @classmethod
    def create_dat_icons(cls):
        from ..preferences import prefs
        base_path = Path(__file__).resolve().parent
        filename = base_path / ('univ_mono.dat' if prefs().color_mode == 'MONO' else 'univ.dat')
        with open(filename, 'wb') as file:
            fw = file.write
            # Header (version 0).
            fw(b'VCO\x00')
            # Width, Height
            fw(bytes((255, 255)))
            # X, Y
            fw(bytes((0, 0)))
            icon_flat_data = cls.create_batch_from_tris_and_colors()
            fw(icon_flat_data)

        cls.update_wst_icon(str(filename))

    @staticmethod
    def update_wst_icon(filename: str):
        """ Icons for WST are cached during class registration and remain even after unregister.
        To update an icon, the old one must be removed and the new one registered.

        This hack is used to avoid restarting Blender."""

        from .. import utils
        from ..ui import UNIV_WT_object_VIEW3D
        from bl_ui.space_toolsystem_common import _icon_cache as wst_icons_cache  # noqa

        icon_name = UNIV_WT_object_VIEW3D.bl_icon
        if icon_name in wst_icons_cache:
            if wst_icons_cache[icon_name] != 0:
                bpy.app.icons.release(wst_icons_cache[icon_name])

            try:
                icon_value = bpy.app.icons.new_triangles_from_file(filename)
            except Exception as e:  # noqa
                import traceback
                traceback.print_exc()
                print(f"UniV: WS Tool icon could not be reloaded from {filename!r}")
                icon_value = 0
            wst_icons_cache[icon_name] = icon_value
            utils.update_area_by_type('VIEW_3D')


class IconsCreator:
    """
    1) This class imports SVG files
    2) Converts them to MESH objects
    3) Extracts mesh draw data (triangles and their colors)
    4) Renders to a buffer
    5) Reads from the buffer and passes it to OpenImageIO
    6) OIIO performs downscaling to reduce aliasing and saves as .png

    To ensure crisp icons, shape edges should align exactly to pixel boundaries wherever possible,
    and transforms should be removed to reduce floating-point errors.
    The Apply Transform add-on in Inkscape can help with this, though it doesn’t always give accurate results,
    so achieving high quality often requires a lot of manual work."""

    @classmethod
    def convert_svg_to_png_builtin(cls, icon_size=32, mono=False, antialiasing=2):
        base_path = Path(__file__).resolve().parent
        svg_folder = base_path / ('svg_mono' if mono else 'svg')
        png_folder = base_path / ('png_mono' if mono else 'png')

        prev_data = PreviousData()
        from io_curve_svg import import_svg
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')

        for attr in dir(icons):
            if not attr.endswith('_'):
                if not isinstance(getattr(icons, attr), int):
                    raise AttributeError(f"{attr!r} is not a valid icon attribute")

                svg_file = svg_folder / f"{attr}.svg"
                if not svg_file.exists():
                    print(f'UniV: File {svg_file} not found')
                    continue
                png_save_path = png_folder / f"{attr}.png"

                import_svg.load(None, bpy.context, filepath=str(svg_file))

                svg_objects = list(set(bpy.context.scene.objects) - prev_data.prev_objects)
                if not svg_objects:
                    raise AttributeError(f'{attr!r} not have objects')

                bpy.context.view_layer.objects.active = svg_objects[0]
                for svg_obj in svg_objects:
                    assert svg_obj.type == 'CURVE'
                    svg_obj.select_set(True)
                bpy.ops.object.convert(target='MESH', keep_original=False)

                mesh_objects = list(set(bpy.context.scene.objects) - prev_data.prev_objects)
                tris, colors = cls.calc_tris_for_draw(mesh_objects, attr)  # extract draw data

                offscreen = gpu.types.GPUOffScreen(icon_size * antialiasing, icon_size * antialiasing)  # noqa
                offscreen.bind()

                try:
                    fb = gpu.state.active_framebuffer_get()
                    fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                    cls.draw_image(tris, colors, 32)

                    pixel_data = fb.read_color(0, 0, icon_size * antialiasing, icon_size * antialiasing, 4, 0, 'UBYTE')
                    pixel_data.dimensions = (icon_size * antialiasing) * (icon_size * antialiasing) * 4
                    cls.save_pixels(str(png_save_path), pixel_data, icon_size, icon_size, antialiasing)
                finally:
                    offscreen.unbind()
                    offscreen.free()

                for obj in svg_objects:
                    bpy.data.objects.remove(obj)

        prev_data.restore()

    # TODO: Implement flat vertices with indexes (with index offset)
    @classmethod
    def calc_tris_for_draw(cls, mesh_objects: list[bpy.types.Object], icon_name: str):
        all_tris = []
        all_colors = []
        for obj in mesh_objects:
            corners_len = len(obj.data.loops)
            if not corners_len:
                continue

            mesh = obj.data
            num_verts = len(mesh.vertices)
            verts_np = np.empty(num_verts * 3, dtype=np.float32)
            mesh.vertices.foreach_get("co", verts_np)
            verts_np = verts_np.reshape(num_verts, 3)

            mesh.calc_loop_triangles()
            num_tris = len(mesh.loop_triangles)
            tris_np = np.empty(num_tris * 3, dtype=np.int32)
            mesh.loop_triangles.foreach_get("vertices", tris_np)
            tris_np = tris_np.reshape(num_tris, 3)

            tri_coords = verts_np[tris_np][:, :, :2]
            tri_coords = tri_coords.reshape(num_tris*3, 2)
            all_tris.append(tri_coords)

            all_colors.extend([cls.get_color(obj, icon_name)] * len(tri_coords))

        assert all_tris
        all_tris = np.concatenate(all_tris, axis=0)
        return all_tris, all_colors

    @classmethod
    def draw_image(cls, coords, colors, icon_size):
        gpu.state.blend_set('ALPHA')

        with gpu.matrix.push_pop():
            gpu.matrix.load_matrix(cls.get_normalize_uvs_matrix(icon_size))
            gpu.matrix.load_projection_matrix(Matrix.Identity(4))
            cls.draw_background_colors(coords, colors)
        gpu.state.blend_set('NONE')

    @classmethod
    def get_normalize_uvs_matrix(cls, icon_size):
        """Matrix maps x and y coordinates from [0, 1] to [-1, 1]"""
        matrix = Matrix.Identity(4)
        matrix.col[3][0] = -1
        matrix.col[3][1] = 1
        matrix[0][0] = 2
        matrix[1][1] = -2

        # NOTE: The operation order is important to match the SVG converter's float precision quirks - changing it would break proportions.
        svg_matrix = Matrix()
        svg_matrix = svg_matrix @ Matrix.Scale(1.0 / 90.0 * 0.3048 / 12.0, 4, Vector((1.0, 0.0, 0.0)))
        svg_matrix = svg_matrix @ Matrix.Scale(-1.0 / 90.0 * 0.3048 / 12.0, 4, Vector((0.0, 1.0, 0.0)))
        svg_objects_dimension = svg_matrix @ Vector((icon_size, icon_size, 0))

        filled_scale = [1 / abs(component) for component in svg_objects_dimension.xy]
        filled_scale.append(1)
        fit_matrix = Matrix.Diagonal(filled_scale).to_4x4()

        return matrix @ fit_matrix

    @classmethod
    def save_pixels(cls, filepath, pixel_data, width, height, antialiasing):
        import OpenImageIO as oiio
        spec = oiio.ImageSpec(width, height, 4, 'uint8')
        # https://github.com/AcademySoftwareFoundation/OpenImageIO/blob/main/src/png.imageio/pngoutput.cpp
        spec.attribute('png:compressionLevel', 9)
        spec_aa = oiio.ImageSpec(width*antialiasing, height*antialiasing, 4, 'uint8')

        buf_extended = oiio.ImageBuf(spec_aa)
        buf_resized = oiio.ImageBuf(spec)
        buf_extended.set_pixels(oiio.ROI(0, width*antialiasing, 0, height*antialiasing), pixel_data)
        # NOTE: Resize gives a better result than resample
        oiio.ImageBufAlgo.resize(buf_resized, buf_extended)
        buf_resized.write(filepath)

    @classmethod
    def draw_background_colors(cls, coords, colors):
        shader = gpu.shader.from_builtin('FLAT_COLOR')
        from gpu_extras.batch import batch_for_shader
        batch = batch_for_shader(
            shader, 'TRIS',
            {"pos": coords, "color": colors})
        batch.draw(shader)
    
    @staticmethod
    def get_color(obj, icon_name):
        """
        This is where color matching happens.
        To ensure correct results, you’ll need to download/open Inkscape, open your icon, then use Save As -> Optimized SVG.
        This removes metadata, extra characters, and converts RGB to HEX (if the corresponding options are enabled).
        """
        from ..preferences import prefs
        from ..utils import hex_to_rgb, vec_isclose

        linear_color = obj.active_material.diffuse_color[:]
        srgb_color = Color(linear_color[:3]).from_scene_linear_to_srgb()
        ret_color = *srgb_color, linear_color[-1]
        def linear_to_ret_color(lin_color):
            srgb_color_ = Color(lin_color[:3]).from_scene_linear_to_srgb()
            return *srgb_color_, lin_color[-1]

        found_color = True
        if vec_isclose(srgb_color, hex_to_rgb('#ffffff'), 0.001):  # White
            linear_white = prefs().icon_white_color
            ret_color = linear_to_ret_color(linear_white)
        elif vec_isclose(srgb_color, hex_to_rgb('#ececec'), 0.001):  # Select Arrow
            linear_select_arrow = prefs().icon_select_arrow_color
            ret_color = linear_to_ret_color(linear_select_arrow)

        elif prefs().color_mode == 'MONO':
            if vec_isclose(srgb_color, hex_to_rgb('#8bc6a1'), 0.001):  # Green
                linear_green = prefs().icon_mono_green
                ret_color = linear_to_ret_color(linear_green)
            elif vec_isclose(srgb_color, hex_to_rgb('#c7c7c7'), 0.001):  # Grey
                linear_gray = prefs().icon_mono_gray
                ret_color = linear_to_ret_color(linear_gray)
            else:
                found_color = False
        else:
            if vec_isclose(srgb_color, hex_to_rgb('#7d87ff'), 0.001):  # Violet
                linear_violet = prefs().icon_colored_violet
                ret_color = linear_to_ret_color(linear_violet)
            elif vec_isclose(srgb_color, hex_to_rgb('#62cdf9'), 0.001):  # Cian
                linear_cian = prefs().icon_colored_cian
                ret_color = linear_to_ret_color(linear_cian)

            elif vec_isclose(srgb_color, hex_to_rgb('#dc87ff'), 0.001):  # Purple
                linear_purple = prefs().icon_colored_purple
                ret_color = linear_to_ret_color(linear_purple)
            elif vec_isclose(srgb_color, hex_to_rgb('#ff87a9'), 0.001):  # Pink
                linear_pink = prefs().icon_colored_pink
                ret_color = linear_to_ret_color(linear_pink)
            else:
                found_color = False

        if not found_color:
            from ..utils import rgb_to_hex
            print(f'UniV: Generate Icons: Not found color {rgb_to_hex(srgb_color)!r} for icon {icon_name!r}')

        return ret_color

class UNIV_OT_IconsGenerator(bpy.types.Operator):
    bl_idname = 'wm.univ_icons_generator'
    bl_label = 'Generate'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = ("The Workspace Tool shader uses slightly different parameters, so colors may vary. "
                      "It's recommended to adjust them manually for best results.")

    generate_only_ws_tool_icon: bpy.props.BoolProperty(name='Tool Icons', default=False)
    def execute(self, context):
        if self.generate_only_ws_tool_icon:
            WSToolIconsGenerator.create_dat_icons()
        else:
            from ..preferences import prefs
            IconsCreator.convert_svg_to_png_builtin(
                icon_size=int(prefs().icon_size),
                mono=prefs().color_mode == 'MONO',
                antialiasing=int(prefs().icon_antialiasing)
            )
            icons.unregister_icons_()
            icons.register_icons_()
            WSToolIconsGenerator.create_dat_icons()
        return {'FINISHED'}
