
import bpy

from .. import types
from .. import utils

# Patterns for future
# _ARS-10.5  # Arrow Scale
# _C(1,2,3)-FFFFFF_  # Color
# _P1-10.5   # Pattern 1
# _LW-2     # Line Width PX
# _SC-2     # Scale
# _SHP      # Shape
# UniV_ColorGrid_2K_
# UniV_Grid_2Kx512_

generated_types = (
    ('UV_GRID', 'Grid', ''),
    ('COLOR_GRID', 'Color Grid', ''),
)

class UNIV_OT_Checker(bpy.types.Operator):
    bl_idname = "mesh.univ_checker"
    bl_label = "Checker Map"
    bl_description = "Used as a texture for testing UV maps"
    bl_options = {'REGISTER', 'UNDO'}

    generated_type: bpy.props.EnumProperty(name='Texture', default='UV_GRID', items=generated_types)
    # show_in_2d: bpy.props.BoolProperty(name='Show in 2D View', default=True)
    size_x: bpy.props.EnumProperty(name='X', default='2K', items=utils.resolutions,
                                   update=lambda self, _: setattr(self, 'size_y', self.size_x) if self.lock_size and self.size_x != self.size_y else None)
    size_y: bpy.props.EnumProperty(name='Y', default='2K', items=utils.resolutions,
                                   update=lambda self, _: setattr(self, 'size_x', self.size_y) if self.lock_size and self.size_x != self.size_y else None)
    lock_size: bpy.props.BoolProperty(name='Lock Size', default=True,
                                      update=lambda self, _: setattr(self, 'size_y', self.size_x) if self.lock_size and self.size_x != self.size_y else None)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def draw(self, context):
        self.layout.row().prop(self, 'generated_type', expand=True)
        self.layout.prop(self, 'lock_size')

        row = self.layout.row(align=True, heading='Size')
        row.prop(self, 'size_x', text='')
        row.prop(self, 'size_y', text='')

    # def invoke(self, context, event):
    #     wm = context.window_manager
    #     return wm.invoke_props_dialog(self)

    def __init__(self):
        self.pattern_name: str = ''
        self.resolution_name: str = ''
        self.full_pattern_name: str = ''
        self.int_size_x: int = -1
        self.int_size_y: int = -1

    def execute(self, context):
        self.pattern_name = self.get_name_from_idname(self.generated_type)
        self.resolution_name: str = self.resolution_str_values_to_name(self.size_x, self.size_y)
        self.full_pattern_name = f"UniV_{self.pattern_name}_{self.resolution_name}"

        self.int_size_x = utils.resolutions_name_by_value[self.size_x]
        self.int_size_y = utils.resolutions_name_by_value[self.size_y]

        mtl = self.get_checker_material()
        node_group = self.get_checker_node_group()
        self.checker_modifier_sanitize()
        self.create_gn_checker_modifier(node_group, mtl)
        self.update_views()
        return {'FINISHED'}

    def get_checker_texture(self) -> bpy.types.Image:
        """Get exist checker texture"""
        for image in reversed(bpy.data.images):
            if (image_name := image.name).startswith(self.full_pattern_name):
                image_width, image_height = image.size
                if self.x_check(image_name) or not image_height or (image_width != self.int_size_x or image_height != self.int_size_y):
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
        """Get exist checker material"""
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
    def checker_modifier_sanitize():
        def remove_not_needed(obj_):
            if obj_.type == 'MESH':
                checker_modifiers_ = []
                for m_ in obj_.modifiers:
                    if isinstance(m_, bpy.types.NodesModifier):
                        if m_.name.startswith('UniV Checker'):
                            if not m_.show_in_editmode:
                                m_.show_in_editmode = True
                            if not m_.show_viewport:
                                m_.show_viewport = True
                            checker_modifiers_.append(m_)
                if len(checker_modifiers_) <= 1:
                    return

                for m_ in checker_modifiers_[:-1]:  # TODO: Save when == pattern
                    obj_.modifiers.remove(m_)

                # Move to bottom
                for idx, m_ in enumerate(obj_.modifiers):
                    if checker_modifiers_[-1] == m_:
                        if len(obj_.modifiers)-1 != idx:
                            obj_.modifiers.move(idx, len(obj_.modifiers))
                        return

        if bpy.context.mode == 'EDIT_MESH':
            for obj in bpy.context.selected_objects:
                if obj.type == 'MESH' and obj.modifiers:
                    checker_modifiers = []
                    for m in obj.modifiers:
                        if isinstance(m, bpy.types.NodesModifier):
                            if m.name.startswith('UniV Checker'):
                                if not m.show_in_editmode:
                                    m.show_in_editmode = True
                                if not m.show_viewport:
                                    m.show_viewport = True
                                checker_modifiers.append(m)
                    if len(checker_modifiers) > 1:
                        break
            else:
                return
            for obj in types.UMeshes.loop_for_object_mode_processing(without_selection=True):
                remove_not_needed(obj)

        else:
            for obj in bpy.context.selected_objects:
                remove_not_needed(obj)

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
    def resolution_name_to_values(name: str):
        if 'x' in name:
            xsize, ysize = name.split('x')
        else:
            xsize = ysize = name
        return utils.resolutions_name_by_value[xsize], utils.resolutions_name_by_value[ysize]

    @staticmethod
    def resolution_values_to_name(xsize: int, ysize: int):
        xsize_name = ysize_name = ''
        if xsize != ysize:
            for k, v in utils.resolutions_name_by_value.items():
                if v == xsize:
                    xsize_name = k
                if v == ysize:
                    ysize_name = k
            assert (xsize_name and ysize_name), f'Not found resolutions {xsize=} {ysize=} in {utils.resolutions_name_by_value}'
            return f'{xsize_name}x{ysize_name}'
        else:
            for k, v in utils.resolutions_name_by_value.items():
                if v == xsize:
                    return k
        assert (xsize_name and ysize_name), f'Not found resolutions {xsize=} {xsize_name=}, {ysize=} {ysize_name=}'

    @staticmethod
    def resolution_str_values_to_name(xsize: str, ysize: str):
        return f'{xsize}x{ysize}' if xsize != ysize else xsize

    @staticmethod
    def get_id_name_from_name(name):
        for gt in generated_types:
            if gt[1] == name:
                return gt[0]
        raise NotImplementedError(f'Texture {name} not implement')

    @staticmethod
    def get_name_from_idname(name):
        for gt in generated_types:
            if gt[0] == name:
                return gt[1]
        raise NotImplementedError(f'Texture {name} not implement')

    def _generate_checker_texture(self):
        full_image_name = f"UniV_{self.get_name_from_idname(self.generated_type)}_{self.resolution_str_values_to_name(self.size_x, self.size_y)}"

        if self.generated_type in ('UV_GRID', 'COLOR_GRID'):
            before = set(bpy.data.images)
            bpy.ops.image.new(name=full_image_name, width=self.int_size_x, height=self.int_size_y, alpha=False, generated_type=self.generated_type)
            return tuple(set(bpy.data.images) - before)[0]
        else:
            raise NotImplementedError(f'Texture {self.generated_type} not implement')

    @staticmethod
    def update_views():
        for area in utils.get_areas_by_type('VIEW_3D'):
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    if space.shading.type == 'SOLID':
                        space.shading.color_type = 'TEXTURE'
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
        self.remove_images()
        return {'FINISHED'}

    @staticmethod
    def remove_modifiers():
        if bpy.context.mode == 'EDIT_MESH':
            selected_objects = types.UMeshes.loop_for_object_mode_processing(without_selection=True)
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
