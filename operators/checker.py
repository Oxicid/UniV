# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

import bpy
import numpy as np

from .. import utypes
from .. import utils
from ..preferences import prefs, checker_generated_types

# Patterns for future
# _ARS-10.5  # Arrow Scale
# _C(1,2,3)-FFFFFF_  # Color
# _P1-10.5   # Pattern 1
# _LW-2     # Line Width PX
# _SC-2     # Scale
# _SHP      # Shape
# UniV_ColorGrid_2K_
# UniV_Grid_2Kx512_


class UNIV_OT_Checker(bpy.types.Operator):
    bl_idname = "mesh.univ_checker"
    bl_label = "Checker"
    bl_description = "Used as a texture for testing UV maps"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True, heading='Apply Method')
        row.scale_x = 0.92
        row.prop(prefs(), 'checker_toggle', expand=True)

        row = layout.row(align=True, heading='Texture Type')
        row.scale_x = 0.92
        row.prop(prefs(), 'checker_generated_type', expand=True)

        row = layout.row(align=True, heading='Size')
        row.prop(prefs(), 'size_x', text='')
        row.prop(prefs(), 'lock_size', text='', icon='LOCKED' if prefs().lock_size else 'UNLOCKED')
        row.prop(prefs(), 'size_y', text='')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_name: str = ''
        self.resolution_name: str = ''
        self.full_pattern_name: str = ''
        self.int_size_x: int = -1
        self.int_size_y: int = -1

    def execute(self, context):
        self.int_size_x = int(prefs().size_x)
        self.int_size_y = int(prefs().size_y)

        self.pattern_name = self.get_name_from_gen_type_idname(prefs().checker_generated_type)
        self.resolution_name: str = self.resolution_values_to_name(self.int_size_x, self.int_size_y)
        self.full_pattern_name = f"UniV_{self.pattern_name}_{self.resolution_name}"

        mtl = self.get_checker_material()

        node_group = self.get_checker_node_group()
        for obj in bpy.context.selected_objects:
            utils.remove_univ_duplicate_modifiers(obj, 'UniV Checker',
                                                  toggle_enable=prefs().checker_toggle == 'TOGGLE')

        def set_active_image():
            if (area := bpy.context.area) and area.type == 'IMAGE_EDITOR':
                for node in mtl.node_tree.nodes:
                    if node.bl_idname == 'ShaderNodeTexImage':
                        assert node.image.name.startswith(self.full_pattern_name)
                        space_data = area.spaces.active
                        space_data.image = node.image
                        break

        if prefs().checker_toggle == 'TOGGLE':
            if self.all_has_enable_gn_checker_modifier():
                self.disable_all_gn_checker_modifier()
            else:
                set_active_image()
                self.enable_and_set_gn_checker_modifier(node_group, mtl)
        else:
            # TODO: Add Overwrite with toggle
            set_active_image()
            self.create_gn_checker_modifier(node_group, mtl)
        self.update_views()
        return {'FINISHED'}

    def get_checker_texture(self) -> bpy.types.Image:
        """Get exist checker texture"""
        for image in reversed(bpy.data.images):
            if (image_name := image.name).startswith(self.full_pattern_name):
                image_width, image_height = image.size
                if self.x_check(image_name) or not image_height or (
                        image_width != self.int_size_x or image_height != self.int_size_y):
                    if image.users == 0:
                        bpy.data.images.remove(image)
                        print(f"UniV: Checker Image '{image_name}' was removed")
                    continue

                return image
        return self._generate_checker_texture()

    def material_is_changed(self, mtl):
        if mtl.use_nodes and len(nodes := mtl.node_tree.nodes) == 3:
            if output_node := [n for n in nodes if n.bl_idname == 'ShaderNodeOutputMaterial']:
                if output_node[0].target == 'ALL' and output_node[0].inputs[0].links:
                    diffuse = output_node[0].inputs[0].links[0].from_node
                    if not diffuse.mute and diffuse.bl_idname == 'ShaderNodeBsdfDiffuse':
                        if diffuse.inputs[0].links:
                            img_node = diffuse.inputs[0].links[0].from_node
                            if not img_node.mute and img_node.bl_idname == 'ShaderNodeTexImage':
                                img = img_node.image
                                if img and img.name.startswith(self.full_pattern_name) and not self.x_check(img.name):
                                    if img.size[0] == self.int_size_x and img.size[1] == self.int_size_y:
                                        return False
        return True

    def get_checker_material(self) -> bpy.types.Material | None:
        """Get exist checker material"""
        for mtl in reversed(bpy.data.materials):
            if (mtl_name := mtl.name).startswith(self.full_pattern_name):
                if self.x_check(mtl_name) or self.material_is_changed(mtl):
                    if mtl.users == 0:
                        bpy.data.materials.remove(mtl)
                        print(f"UniV: Checker Material '{mtl_name}' was removed")
                else:
                    return mtl
        return self._create_checker_material()

    def _create_checker_material(self):

        img = self.get_checker_texture()

        mtl = bpy.data.materials.new(name=self.full_pattern_name)
        mtl.use_nodes = True

        nodes = mtl.node_tree.nodes
        nodes.clear()

        output = nodes.new(type="ShaderNodeOutputMaterial")
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        node_image = nodes.new(type="ShaderNodeTexImage")

        output.location = (200, 0)
        diffuse.location = (0, 0)
        node_image.location = (-300, 0)

        mtl.node_tree.links.new(diffuse.outputs[0], output.inputs[0])
        mtl.node_tree.links.new(node_image.outputs[0], diffuse.inputs[0])

        node_image.image = img

        nodes.active = node_image
        return mtl

    @staticmethod
    def checker_node_group_is_changed(node_group):
        if len(nodes := node_group.nodes) == 3:
            if output_node := [n for n in nodes if n.bl_idname == 'NodeGroupOutput']:
                if output_node[0].inputs and (output_links := output_node[0].inputs[0].links):
                    if (set_material_node := output_links[0].from_node).bl_idname == 'GeometryNodeSetMaterial':
                        geometry_links = set_material_node.inputs[0].links
                        if geometry_links and geometry_links[0].from_node.bl_idname == 'NodeGroupInput':
                            return False
        return True

    def get_checker_node_group(self):
        """Get exist checker node group"""
        for ng in reversed(bpy.data.node_groups):
            if ng.name.startswith('UniV Checker'):
                if self.checker_node_group_is_changed(ng):
                    if ng.users == 0:
                        bpy.data.node_groups.remove(ng)
                else:
                    return ng
        return self._create_checker_node_group()

    @staticmethod
    def _create_checker_node_group():
        node_group = bpy.data.node_groups.new(name='UniV Checker', type='GeometryNodeTree')

        input_node = node_group.nodes.new(type="NodeGroupInput")
        output_node = node_group.nodes.new(type="NodeGroupOutput")

        input_node.location = (-200, 0)
        output_node.location = (200, 0)

        if iface := getattr(node_group, 'interface', None):
            iface.new_socket('Input', description="", in_out='INPUT', socket_type='NodeSocketGeometry')
            iface.new_socket('Checker Material', description="", in_out='INPUT', socket_type='NodeSocketMaterial')
            iface.new_socket('Output', description="", in_out='OUTPUT', socket_type='NodeSocketGeometry')
        else:
            node_group.inputs.new('NodeSocketGeometry', 'Input')
            node_group.inputs.new('NodeSocketMaterial', 'Checker Material')
            node_group.outputs.new('NodeSocketGeometry', 'Output')

        set_material_node = node_group.nodes.new(type="GeometryNodeSetMaterial")
        set_material_node.location = (0, 0)

        mtl_socket = [s for s in set_material_node.inputs if s.bl_idname == 'NodeSocketMaterial'][0]

        node_group.links.new(input_node.outputs['Input'], set_material_node.inputs['Geometry'])
        node_group.links.new(set_material_node.outputs['Geometry'], output_node.inputs['Output'])
        node_group.links.new(input_node.outputs['Checker Material'], mtl_socket)

        return node_group

    @staticmethod
    def create_gn_checker_modifier(node_group, mtl):
        for obj in bpy.context.selected_objects:
            if not obj.type == 'MESH':
                continue
            has_checker_modifier = False
            for m in obj.modifiers:
                if not isinstance(m, bpy.types.NodesModifier):
                    continue
                if m.name.startswith('UniV Checker'):
                    has_checker_modifier = True
                    if m.node_group != node_group:
                        m.node_group = node_group
                    if 'Socket_1' in m:
                        if m['Socket_1'] != mtl:
                            m['Socket_1'] = mtl
                    else:
                        # old version support (version???)
                        if m['Input_1'] != mtl:
                            m['Input_1'] = mtl
                    obj.update_tag()
                    break
            if not has_checker_modifier:
                m = obj.modifiers.new(name='UniV Checker', type='NODES')
                m.node_group = node_group
                m.show_render = False
                if 'Socket_1' in m:
                    m['Socket_1'] = mtl
                else:
                    m['Input_1'] = mtl

    @staticmethod
    def all_has_enable_gn_checker_modifier():
        counter = 0
        for obj in (selected_objects := [obj_ for obj_ in bpy.context.selected_objects if obj_.type == 'MESH']):
            for m in obj.modifiers:
                if isinstance(m, bpy.types.NodesModifier):
                    if m.name.startswith('UniV Checker'):
                        counter += (m.show_in_editmode and m.show_viewport)
                        break
        return len(selected_objects) == counter

    @staticmethod
    def enable_and_set_gn_checker_modifier(node_group, mtl):
        for obj in bpy.context.selected_objects:
            if not obj.type == 'MESH':
                continue
            has_checker_modifier = False
            for m in obj.modifiers:
                if not isinstance(m, bpy.types.NodesModifier):
                    continue
                if m.name.startswith('UniV Checker'):
                    if not m.show_in_editmode:
                        m.show_in_editmode = True
                    if not m.show_viewport:
                        m.show_viewport = True
                    has_checker_modifier = True
                    obj.update_tag()
                    break
            if not has_checker_modifier:
                m = obj.modifiers.new(name='UniV Checker', type='NODES')
                m.node_group = node_group
                m.show_render = False
                if 'Socket_1' in m:
                    m['Socket_1'] = mtl
                else:
                    m['Input_1'] = mtl

    @staticmethod
    def disable_all_gn_checker_modifier():
        for obj in bpy.context.selected_objects:
            if obj.type == 'MESH':
                for m in obj.modifiers:
                    if isinstance(m, bpy.types.NodesModifier):
                        if m.name.startswith('UniV Checker'):
                            m.show_in_editmode = False
                            m.show_viewport = False
                            break

    @staticmethod
    def resolution_values_to_name(xsize: int, ysize: int):
        x_size_name = utils.resolution_value_to_name[xsize]
        y_size_name = utils.resolution_value_to_name[ysize]
        return f'{x_size_name}x{y_size_name}' if xsize != ysize else x_size_name

    @staticmethod
    def get_name_from_gen_type_idname(name):
        for gt in checker_generated_types:
            if gt[0] == name:
                return gt[1]
        raise NotImplementedError(f'Texture {name} not implement')

    def _generate_checker_texture(self):
        idname = self.get_name_from_gen_type_idname(prefs().checker_generated_type)
        res_name = self.resolution_values_to_name(self.int_size_x, self.int_size_y)
        full_image_name = f"UniV_{idname}_{res_name}"

        if prefs().checker_generated_type in ('UV_GRID', 'COLOR_GRID'):
            before = set(bpy.data.images)
            bpy.ops.image.new(
                name=full_image_name,
                width=self.int_size_x,
                height=self.int_size_y,
                alpha=False,
                generated_type=prefs().checker_generated_type)
            return tuple(set(bpy.data.images) - before)[0]
        else:
            raise NotImplementedError(f'Texture {prefs().checker_generated_type} not implement')

    @staticmethod
    def update_views():
        changed = False
        for area in utils.get_areas_by_type('VIEW_3D'):
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    if space.shading.type == 'SOLID':
                        if space.shading.color_type != 'TEXTURE':
                            space.shading.color_type = 'TEXTURE'
                            changed = True
                    elif space.shading.type == 'WIREFRAME':
                        space.shading.type = 'SOLID'
                        space.shading.color_type = 'TEXTURE'
                        changed = True
        if changed:
            bpy.context.view_layer.update()

    def x_check(self, name):
        """Has 'x' after resolution"""
        return len(name) > (x_idx := len(self.full_pattern_name)) and name[x_idx] == 'x'


class UNIV_OT_CheckerCleanup(bpy.types.Operator):
    bl_idname = "wm.univ_checker_cleanup"
    bl_label = "Checker Map Cleanup"
    bl_description = "Cleanup textures, materials, nodes and modifiers"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        self.remove_modifiers()
        self.remove_node_group()
        self.remove_materials()
        # TODO: Close img from selected meshes
        for a in utils.get_areas_by_type('IMAGE_EDITOR'):
            space = a.spaces.active
            if (img := space.image) and img.name.startswith('UniV_'):  # noqa
                space.image = None
        self.remove_images()
        return {'FINISHED'}

    @staticmethod
    def remove_modifiers():
        if bpy.context.mode == 'EDIT_MESH':
            selected_objects = utypes.UMeshes.loop_for_object_mode_processing(without_selection=True)
        else:
            selected_objects = bpy.context.selected_objects
        # mod_counter = 0
        for obj in selected_objects:
            if obj.type == 'MESH':
                for m in reversed(obj.modifiers):
                    if isinstance(m, bpy.types.NodesModifier):
                        if m.name.startswith('UniV Checker'):
                            obj.modifiers.remove(m)

    @staticmethod
    def remove_node_group():
        for ng in reversed(bpy.data.node_groups):
            if ng.name.startswith('UniV Checker'):
                if not ng.users:
                    bpy.data.node_groups.remove(ng)

    @staticmethod
    def remove_materials():
        for mtl in reversed(bpy.data.materials):
            if mtl.name.startswith('UniV_') and not mtl.users:
                bpy.data.materials.remove(mtl)

    @staticmethod
    def remove_images():
        for img in reversed(bpy.data.images):
            if img.name.startswith('UniV_') and not img.users:
                bpy.data.images.remove(img)


class TexturePatterns:
    @staticmethod
    def draw_lines(texture: utypes.UTexture, step=32, thickness=1, exclude_first=False):
        texture = texture.texture
        first = step if exclude_first else 0
        for x in range(first, texture.width, step):
            texture.draw_vline_wrapped(x, thickness)

        for y in range(first, texture.height, step):
            texture.draw_hline_wrapped(y, thickness)

    @staticmethod
    def draw_dashed_hline(tex: utypes.UTexture, y, dash=6, gap=4, thickness=1, phase=0):
        x = np.arange(tex.width)
        mask = ((x + phase) % (dash + gap)) < dash

        half = thickness // 2
        y0 = y - half
        y1 = y + half + (thickness & 1)

        for ys in tex.wrap_slices_h(y0, y1):
            tex.texture[ys, mask] = True


    @staticmethod
    def draw_dashed_vline(tex: utypes.UTexture, x, dash=6, gap=4, thickness=1, phase=0):
        y = np.arange(tex.height)
        mask = ((y + phase) % (dash + gap)) < dash

        half = thickness // 2
        x0 = x - half
        x1 = x + half + (thickness & 1)

        for xs in tex.wrap_slices_w(x0, x1):
            tex.texture[mask, xs] = True

    @classmethod
    def draw_pluses(cls, texture: utypes.UTexture, step: int=128, size: int=32, thickness: int=3, center=True):
        for x, y in utils.grid_points_px(texture.width, texture.height, step, center):
            cls.draw_plus(texture, y, x, size=size, thickness=thickness)

    @staticmethod
    def draw_plus(tex: utypes.UTexture, cy: int, cx: int, size: int, thickness: int):
        half_size = size // 2
        half_th = thickness // 2

        # Vertical
        tex.fill_wrapped_rect(
            cy - half_size,
            cy + half_size,
            cx - half_th,
            cx + half_th + 1
        )

        if size == thickness:
            return

        # Horizontal
        tex.fill_wrapped_rect(
            cy - half_th,
            cy + half_th + 1,
            cx - half_size,
            cx + half_size
        )


    @classmethod
    def draw_checker(cls, mask, step=32):
        y = np.arange(mask.height)[:, None]
        x = np.arange(mask.width)[None, :]

        mask.texture[:] = ((x // step + y // step) & 1) == 0


    @staticmethod
    def checker_board_text(width: int, height: int, step: int = 128, outline: int = 1):
        import blf
        mono: int = 0 #  blf_mono_font_render;
        text_size = 54
        utils.blf_size(mono, text_size)  # hard coded size!

        # Using nullptr will assume the byte buffer has sRGB colorspace, which currently
        # matches the default colorspace of new images.

        text_color: list[float] = [0.0, 0.0, 0.0, 1.0]
        text_outline: list[float] = [1.0, 1.0, 1.0, 1.0]

        import string
        letters = string.ascii_uppercase
        digits = string.digits[1:] + 'ABCDEF'

        first_char_index: int = 0
        for y in range(0, height, step):
            first_char = letters[first_char_index]

            second_char_index: int = 0
            for x in range(0, width, step):
                second_char = digits[second_char_index]
                text = first_char + second_char
                size_x, size_y = blf.dimensions(mono, text)

                # hard coded offset
                pen_x: int = x + ((step // 2) - (size_x // 2))
                pen_y: int = y + ((step // 2) - (size_y // 2))

                # terribly crappy outline font!
                blf.color(mono, *text_outline)

                for dx, dy in utils.padding_deltas(outline):

                    blf.position(mono, pen_x+dx, pen_y+dy, 0.0)
                    blf.draw_buffer(mono, text)

                blf.color(mono, *text_color)
                blf.position(mono, pen_x, pen_y, 0.0)
                blf.draw_buffer(mono, text)

                second_char_index = (second_char_index + 1) % len(digits)

            first_char_index = (first_char_index + 1) % len(letters)



class UNIV_OT_CheckerTest(bpy.types.Operator, TexturePatterns):
    bl_idname = 'uv.univ_checker_test'
    bl_label = 'Checker Test'
    bl_options = {'REGISTER', 'UNDO'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import gpu
        self.area: bpy.types.Area | None = None
        self.handler = None
        self.batch: gpu.types.GPUBatch | None = None
        self.shader: gpu.types.GPUShader | None = None
        self.size_mul = 2
        self.image_size = (256*self.size_mul, 256*self.size_mul)
        self.offscreen = gpu.types.GPUOffScreen(*self.image_size)



    def invoke(self, context, event):
        try:
            self.area = context.area
            self.register_draw()
            import gpu
            import blf
            import imbuf
            from mathutils import Matrix
            from gpu.types import GPUTexture

            from gpu_extras.presets import draw_circle_2d
            from ..univ_pro import trim
            from .. import draw
            self.compile_shader()

            # Create and fill offscreen
            ##########################################

            width, height = self.image_size

            # ibuf = imbuf.new(self.image_size)

            # font_id = blf.load("/path/to/font.ttf")
            # font_id = 0

            # with blf.bind_imbuf(font_id, ibuf, display_name="sRGB"):
            #     self.checker_board_text(width, height)

            # texture = utypes.UTexture.from_ibuf(ibuf).to_gpu_texture()
            mask = utypes.UTexture(width, height, texture_type=bool)

            # color2 = np.array([1, 0, 1, 1], dtype=np.float32)

            self.draw_pluses(mask, thickness=32, center=False)


            # texture2 = utypes.UTexture(width, height)
            # texture2[mask] = color2

            # mask2 = utypes.UTexture(width, height, texture_type=bool)
            # self.draw_checker(mask, step=1)

            # texture2 = mask2.mask_to_texture([1,1,1,0], [0,0,1,1]).to_gpu_texture()

            # texture2 = mask2.to_gpu_texture()

            # self.draw_dashed_vline(mask,0, thickness=7)
            # self.draw_lines(mask, step=128, thickness=1, exclude_first=False)
            texture2 = mask.mask_to_texture([0.2,0.2,0.2,1], [1.25,0.25,0.25,1.0])

            # mask = utypes.UTexture(width, height, texture_type=bool)
            # self.draw_lines(mask, step=128, exclude_first=False)
            #
            # texture2[mask] = [0.5,0.5,0.5,1.0]
            texture2 = texture2.to_gpu_texture()


            draw.shaders.blend_set_alpha()
            with self.offscreen.bind():
                fb = gpu.state.active_framebuffer_get()
                fb.clear(color=(0.0, 0.0, 0.0, 0.0))
                with gpu.matrix.push_pop():
                    # self.shader.uniform_float("viewProjectionMatrix", self.get_normalize_uvs_matrix())
                    # self.shader.uniform_sampler("image", texture)
                    # self.batch.draw(self.shader)

                    self.shader.uniform_float("viewProjectionMatrix", self.get_normalize_uvs_matrix())
                    self.shader.uniform_sampler("image", texture2)
                    self.batch.draw(self.shader)

                    # Reset matrices -> use normalized device coordinates [-1, 1].
                    gpu.matrix.load_matrix(Matrix.Identity(4))
                    gpu.matrix.load_projection_matrix(Matrix.Identity(4))

                    # amount = 10
                    # for i in range(-amount, amount + 1):
                    #     x_pos = i / amount
                        # draw_circle_2d((x_pos, 0.0), (1, 1, 1, 1), 0.5, segments=200)



            img = bpy.data.images.new(
                "temp_text",
                width=width,
                height=height,
                alpha=True,
                float_buffer=True,
            )

            img.pixels = utypes.UTexture.from_frame_buf(width, height, fb).texture.reshape((-1,))
            img.update()

            wm = context.window_manager
            wm.modal_handler_add(self)
            return {'RUNNING_MODAL'}
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}


    def compile_shader(self):
        import gpu
        from gpu_extras.batch import batch_for_shader
        # Drawing the generated texture in 3D space
        #############################################
        vert_out = gpu.types.GPUStageInterfaceInfo("my_interface")
        vert_out.smooth('VEC2', "uvInterp")
        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "viewProjectionMatrix")
        # shader_info.push_constant('MAT4', "modelMatrix")
        shader_info.sampler(0, 'FLOAT_2D', "image")
        shader_info.vertex_in(0, 'VEC2', "position")
        shader_info.vertex_in(1, 'VEC2', "uv")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "FragColor")
        shader_info.vertex_source(
            "void main()"
            "{"
            "  uvInterp = uv;"
            "  gl_Position = viewProjectionMatrix * vec4(position, 0.0, 1.0);"
            "}"
        )
        shader_info.fragment_source(
            "void main()"
            "{"
            "  FragColor = texture(image, uvInterp);"
            "}"
        )
        self.shader = gpu.shader.create_from_info(shader_info)
        del vert_out
        del shader_info
        self.batch = batch_for_shader(
            self.shader, 'TRI_FAN',
            {
                "position": (
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (1.0, 1.0),
                    (0.0, 1.0),
                ),
                "uv": ((0, 0), (1, 0), (1, 1), (0, 1)),
            },
        )

    def modal(self, context, event):
        try:
            return self.modal_ex(context, event)
        except Exception as e:  # noqa
            import traceback
            traceback.print_exc()
            self.report({'ERROR'}, str(e))
            self.exit()
            return {'FINISHED'}


    def modal_ex(self, _context, event):
        if event.type in ('INBETWEEN_MOUSEMOVE', 'TIMER_REPORT'):
            return {'RUNNING_MODAL'}

        elif event.value == 'PRESS':
            if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'MIDDLEMOUSE'}:
                return {'PASS_THROUGH'}

            elif event.type in {'ESC', 'SPACE', 'RET', 'NUMPAD_ENTER', 'RIGHTMOUSE'}:
                return self.exit()

        return {'RUNNING_MODAL'}

    def exit(self):
        for area in utils.get_areas_by_type('IMAGE_EDITOR'):
            area.tag_redraw()

        if not (self.handler is None):
            bpy.types.SpaceImageEditor.draw_handler_remove(self.handler, 'WINDOW')
        return {'FINISHED'}

    def register_draw(self):
        self.handler = bpy.types.SpaceImageEditor.draw_handler_add(
            self.univ_quick_checker_draw_callback, (), 'WINDOW', 'POST_VIEW')
        self.area.tag_redraw()

    def univ_quick_checker_draw_callback(self):
        import gpu
        from .. import draw

        draw.shaders.blend_set_alpha()
        mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
        self.shader.uniform_float("viewProjectionMatrix", mvp)
        self.shader.uniform_sampler("image", self.offscreen.texture_color)
        self.batch.draw(self.shader)

    @staticmethod
    def get_normalize_uvs_matrix():
        """matrix maps x and y coordinates from [0, 1] to [-1, 1]"""
        from mathutils import Matrix
        matrix = Matrix.Identity(4)
        matrix.col[3][0] = -1
        matrix.col[3][1] = -1
        matrix[0][0] = 2
        matrix[1][1] = 2
        return matrix
