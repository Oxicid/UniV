# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later


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