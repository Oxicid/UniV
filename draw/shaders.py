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

# The color doesn't go into the shader, so it's hardcoded as a constant that gets updated when the color changes.
FLAT_SHADING_UNIFORM_COLOR_3D_FOR_UV_FACE_SELECT = None

POLYLINE_UNIFORM_COLOR = None  # Edge Shader with width support
POLYLINE_UNIFORM_COLOR_3D = None  # DEPRECATE for 3.4 and less version

POINT_UNIFORM_COLOR = None  # Edge Shader with width support
POINT_UNIFORM_COLOR_3D = None  # DEPRECATE for 3.4 and less version

set_line_width = lambda width: None
set_line_width_vk = lambda shader, width=2.0: None
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

        if VK_ENABLED:
            global set_line_width_vk

            def _set_line_width_vk(shader, width=2.0):
                shader.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
                shader.uniform_float('lineWidth', width)

            set_line_width_vk = _set_line_width_vk
        else:
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

        cls.init_flat_shading_uniform_color()


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
        return vertex_shader, fragment_shader

    @classmethod
    def init_flat_shading_uniform_color(cls):
        vert_out = gpu.types.GPUStageInterfaceInfo("UniV")
        vert_out.smooth('VEC3', "fcolor")

        shader_info = gpu.types.GPUShaderCreateInfo()
        shader_info.push_constant('MAT4', "mvp")
        shader_info.push_constant('MAT3', "normal_matrix")
        shader_info.push_constant('VEC2', "light_dir")

        shader_info.vertex_in(0, 'VEC3', "pos")
        shader_info.vertex_in(1, 'VEC3', "normal")
        shader_info.vertex_out(vert_out)
        shader_info.fragment_out(0, 'VEC4', "fragColor")

        from .. import preferences
        r, g, b, a = preferences.univ_settings().overlay_3d_uv_face_color
        color_glsl_constant = f"\nconst vec4 color = vec4({r:.6f}, {g:.6f}, {b:.6f}, {a:.6f});\n\n"

        shader_info.vertex_source(color_glsl_constant + '''
        void main()
        {   
            vec3 n = normal_matrix * normal;

            vec4 clip_ = mvp * vec4(pos, 1.0);
            clip_.z -= 0.00005 * clip_.w;
            gl_Position = clip_;
            
            float derive_z = sqrt(1.0f - light_dir.x*light_dir.x - light_dir.y*light_dir.y);
            vec3 light_dir_ = vec3(light_dir, derive_z);
            
            float lambert_light = max(dot(n, light_dir_), 0.3);
            vec3 shaded = color.rgb * lambert_light;

            fcolor = shaded;
        }
        '''
        )

        shader_info.fragment_source(color_glsl_constant + '''
        void main()
        {
            if (gl_FrontFacing) {
                fragColor = vec4(fcolor, color.a);
            } else {
                fragColor = vec4(fcolor * 0.6, color.a);
            }
        }
        '''
        )

        global FLAT_SHADING_UNIFORM_COLOR_3D_FOR_UV_FACE_SELECT
        FLAT_SHADING_UNIFORM_COLOR_3D_FOR_UV_FACE_SELECT = gpu.shader.create_from_info(shader_info)

    @staticmethod
    def unpack_vec4() -> str:
        return  '''
        vec4 unpack_vec4(uint packed) {
            float r = float((packed >> 24) & 0xFF000000u) / 255.0;
            float g = float((packed >> 16) & 0x00FF0000u) / 255.0;
            float b = float((packed >> 8)  & 0x0000FF00u) / 255.0;
            float a = float(packed & 0x000000FF) / 255.0;
            return vec4(r, g, b, a);
        }

        '''
