# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import gpu

VK_ENABLED = False

UNIFORM_COLOR = None
UNIFORM_COLOR_3D = None
FLAT_SHADING_UNIFORM_COLOR_3D = None

POLYLINE_UNIFORM_COLOR = None  # Edge Shader with width support
POLYLINE_UNIFORM_COLOR_3D = None  # DEPRECATE for 3.4 and less version

POINT_UNIFORM_COLOR = None  # Edge Shader with width support
POINT_UNIFORM_COLOR_3D = None  # DEPRECATE for 3.4 and less version

set_line_width = lambda width: None
set_line_width_vk = lambda shader: None
set_point_size = gpu.state.point_size_set

blend_set_alpha = lambda : gpu.state.blend_set('ALPHA')
blend_set_none = lambda : gpu.state.blend_set('NONE')

class Shaders:

    @classmethod
    def init_shaders(cls):
        cls.init_functions()

        global POLYLINE_UNIFORM_COLOR
        if VK_ENABLED:
            POLYLINE_UNIFORM_COLOR = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
        else:
            if bpy.app.version >= (3, 5, 0):
                POLYLINE_UNIFORM_COLOR = gpu.shader.from_builtin('UNIFORM_COLOR')
            else:
                POLYLINE_UNIFORM_COLOR = gpu.shader.from_builtin('2D_UNIFORM_COLOR')


        global POLYLINE_UNIFORM_COLOR_3D
        POLYLINE_UNIFORM_COLOR_3D = POLYLINE_UNIFORM_COLOR
        if bpy.app.version < (3, 5, 0):
            POLYLINE_UNIFORM_COLOR_3D = gpu.shader.from_builtin('3D_UNIFORM_COLOR')


        global POINT_UNIFORM_COLOR
        if VK_ENABLED:
            POINT_UNIFORM_COLOR = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
        else:
            POINT_UNIFORM_COLOR = gpu.shader.from_builtin(
                '2D_UNIFORM_COLOR' if bpy.app.version < (3, 5, 0) else 'UNIFORM_COLOR')


        global POINT_UNIFORM_COLOR_3D
        POINT_UNIFORM_COLOR_3D = POINT_UNIFORM_COLOR
        if bpy.app.version < (3, 5, 0):
            POINT_UNIFORM_COLOR_3D = gpu.shader.from_builtin('3D_UNIFORM_COLOR')


        global UNIFORM_COLOR
        if bpy.app.version >= (3, 5, 0):
            UNIFORM_COLOR = gpu.shader.from_builtin('UNIFORM_COLOR')
        else:
            UNIFORM_COLOR = gpu.shader.from_builtin('2D_UNIFORM_COLOR')


        global UNIFORM_COLOR_3D
        UNIFORM_COLOR_3D = UNIFORM_COLOR
        if bpy.app.version < (3, 5, 0):
            POINT_UNIFORM_COLOR_3D = gpu.shader.from_builtin('3D_UNIFORM_COLOR')


    @classmethod
    def init_functions(cls):
        global VK_ENABLED
        if draw_engine_type := getattr(gpu.platform, "backend_type_get", None):
            VK_ENABLED = draw_engine_type() == "VULKAN"

        if not VK_ENABLED:
            global set_line_width
            global set_point_size

            global blend_set_alpha
            global blend_set_none

            if bpy.app.version >= (3, 5, 0):
                set_line_width = gpu.state.line_width_set
            else:
                import bgl
                set_line_width = bgl.glLineWidth
                set_point_size = bgl.glPointSize

                blend_set_alpha = lambda: bgl.glEnable(bgl.GL_ALPHA)
                blend_set_none = lambda: bgl.glDisable(bgl.GL_BLEND)
        else:
            global set_line_width_vk

            def _set_line_width_vk(shader):
                shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
                shader.uniform_float('lineWidth', 2.0)

            set_line_width_vk = _set_line_width_vk


        cls.init_flat_shading_uniform_color()


    @staticmethod
    def init_flat_shading_uniform_color():
        vertex_shader = '''
        in vec3 pos;
        in vec3 normal;
        
        uniform mat4 mvp;
        uniform vec4 color;
        uniform mat3 normal_matrix;
        uniform vec3 light_dir;

        out vec4 fcolor;

        void main()
        {
            vec3 n = normalize(normal_matrix * normal);
            vec3 offset = pos + (n * 0.01);
            gl_Position = mvp * vec4(offset, 1.0);

            float lambert_light = max(dot(n, normalize(light_dir)), 0.3);
            vec3 shaded = color.rgb * lambert_light;

            fcolor = vec4(shaded, color.a);
        }
        '''

        fragment_shader = '''
        in vec4 fcolor;
        out vec4 fragColor;

        void main()
        {
            if (gl_FrontFacing) {
                fragColor = blender_srgb_to_framebuffer_space(fcolor);
            } else {
                fragColor = blender_srgb_to_framebuffer_space(vec4(fcolor.rgb * 1.5, fcolor.a));
            }
        }
        '''
        global FLAT_SHADING_UNIFORM_COLOR_3D
        FLAT_SHADING_UNIFORM_COLOR_3D = gpu.types.GPUShader(vertex_shader, fragment_shader)

    @staticmethod
    def get_round_shape_vertex():
        vertex_shader = """
            uniform mat4 ModelViewProjectionMatrix;
            in vec3 pos;
    
            void main()
            {
                gl_Position = ModelViewProjectionMatrix * vec4(pos, 0.999);
            }
        """
        # draw a round yellow shape to represent a vertex
        fragment_shader = """
            void main()
            {
                float r = 0.0, delta = 0.0, alpha = 0.0;
                vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                r = dot(cxy, cxy);
    
                if (r > 1.0) {
                    discard;
                }
    
                gl_FragColor = vec4(1.0, 1.0, 0.0, 1);
            }
        """

