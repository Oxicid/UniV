"""
Created by Oxicid

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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