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

import bpy

from ..types import Islands
from .. import utils
from ..utils import UMeshes, vec_lerp

class RelaxData:
    def __init__(self, _umesh, _selected_elem, _coords_before, _border_corners, _save_transform_islands):
        self.umesh = _umesh
        self.selected_elem = _selected_elem
        self.coords_before = _coords_before
        self.border_corners = _border_corners
        self.save_transform_islands = _save_transform_islands


class UNIV_OT_Relax(bpy.types.Operator):
    bl_idname = "uv.univ_relax"
    bl_label = "Relax"
    bl_options = {'REGISTER', 'UNDO'}

    iterations: bpy.props.IntProperty(name='Iterations', default=50, min=5, max=600, soft_min=50, soft_max=200)
    border_blend: bpy.props.FloatProperty(name='Border Blend', default=0.1, min=0, soft_min=0, soft_max=1)

    @classmethod
    def poll(cls, context):
        if not context.active_object:
            return False
        if context.active_object.mode != 'EDIT':
            return False
        return True

    def draw(self, context):
        self.layout.prop(self, 'iterations', slider=True)
        self.layout.prop(self, 'border_blend', slider=True)

    def invoke(self, context, event):
        return self.execute(context)

    def __init__(self):
        self.sync: bool = utils.sync()
        self.umeshes: UMeshes | None = None

    def execute(self, context):
        self.umeshes = UMeshes()
        if self.sync:
            if bpy.context.tool_settings.mesh_select_mode[2]:
                self.relax_sync_faces()
            else:
                self.relax_sync_verts_edges()
        else:
            self.relax_non_sync()

        for umesh in self.umeshes:
            umesh.bm.select_flush_mode()
        return self.umeshes.update()

    def relax_sync_verts_edges(self):

        relax_data: list[RelaxData] = []

        for umesh in reversed(self.umeshes):
            if bpy.context.tool_settings.mesh_select_mode[1]:  # EDGE
                if not (selected_elem := utils.calc_selected_edges(umesh)):
                    self.umeshes.umeshes.remove(umesh)
                    continue
            elif bpy.context.tool_settings.mesh_select_mode[0]:  # VERTEX
                if not (selected_elem := utils.calc_selected_verts(umesh)):
                    self.umeshes.umeshes.remove(umesh)
                    continue
            else:
                raise NotImplemented

            uv = umesh.uv_layer
            islands = Islands.calc_visible(umesh.bm, umesh.uv_layer, self.sync)

            for isl in islands:
                isl.mark_seam()

            # Find island border
            border_corners = set()  # Need restore
            for v in umesh.bm.verts:
                if not v.select:
                    continue
                if v.is_boundary:
                    border_corners.update(v.link_loops)
                    continue
                if len({crn[uv].uv.copy().freeze() for crn in v.link_loops}) > 1 or any(not crn.face.select for crn in v.link_loops):
                    border_corners.update(v.link_loops)

            faces_to_select = set()
            verts_to_select = set()

            # Extend selected
            for f in umesh.bm.faces:
                if f.hide or f.select:
                    continue
                if sum(v.select for v in f.verts) not in (0, len(f.verts)):
                    faces_to_select.add(f)
                    for v in f.verts:
                        if not v.select:
                            verts_to_select.add(v)

            for f in faces_to_select:
                f.select = True
            for v in verts_to_select:
                v.select = True

            if bpy.context.tool_settings.mesh_select_mode[1]:  # EDGE
                for e in umesh.bm.edges:
                    e.select = sum(v.select for v in e.verts) == 2

            for f in umesh.bm.faces:
                for crn in f.loops:
                    crn[uv].pin_uv = True

            for crn in border_corners:
                crn[uv].pin_uv = False

            coords_before = [crn[uv].uv.copy() for crn in border_corners]

            save_transform_islands = []
            for isl in islands:
                if any(v.select for f in isl for v in f.verts):
                    save_transform_islands.append(isl.save_transform())

            relax_data.append(RelaxData(umesh, selected_elem, coords_before, border_corners, save_transform_islands))

        self.relax_a(relax_data)

    def relax_a(self, relax_data):
        # Relax
        bpy.ops.uv.minimize_stretch(iterations=self.iterations)
        if any(rd.coords_before for rd in relax_data):
            bpy.ops.uv.unwrap(method='CONFORMAL')
            # Blend Borders
            for rd in relax_data:
                uv = rd.umesh.uv_layer
                for co, crn in zip(rd.coords_before, rd.border_corners):
                    crn_uv = crn[uv]
                    crn_uv.uv = vec_lerp(co, crn_uv.uv, self.border_blend)
        bpy.ops.uv.minimize_stretch(iterations=self.iterations)

        for rd in relax_data:
            for isl in rd.save_transform_islands:  # TODO: Weld half selected islands
                isl.inplace()
        bpy.ops.uv.select_all(action='DESELECT')

        for rd in relax_data:
            for elem in rd.selected_elem:
                elem.select = True

        for rd in relax_data:
            uv = rd.umesh.uv_layer
            for f in rd.umesh.bm.faces:
                for crn in f.loops:
                    crn[uv].pin_uv = False

    def relax_sync_faces(self):
        assert bpy.context.tool_settings.mesh_select_mode[2]
        from ..utils import linked_crn_uv, shared_is_linked

        relax_data: list[RelaxData] = []
        for umesh in reversed(self.umeshes):
            if umesh.is_full_face_deselected:
                self.umeshes.umeshes.remove(umesh)
                continue

            uv = umesh.uv_layer
            islands = Islands.calc_extended(umesh.bm, umesh.uv_layer, self.sync)

            for isl in islands:
                isl.mark_seam()

            to_select = set()
            border_corners = set()

            for f in umesh.bm.faces:
                if not f.select:
                    continue
                for crn in f.loops:
                    linked_crn = linked_crn_uv(crn, uv)
                    linked_crn.append(crn)
                    border = False
                    for _crn in linked_crn:
                        if _crn.face.hide:
                            continue
                        if (next_linked_disc := _crn.link_loop_radial_prev) == _crn:
                            border |= True
                            continue

                        next_face = next_linked_disc.face
                        if next_face.select:
                            continue
                        if next_face.hide:
                            border |= True
                            continue

                        if shared_is_linked(next_linked_disc, _crn, uv):
                            to_select.add(next_linked_disc.face)
                        border |= True
                    if border:
                        border_corners.update(linked_crn)

            for _f in to_select:
                _f.select = True

            for f in umesh.bm.faces:
                for crn in f.loops:
                    crn[uv].pin_uv = True

            for crn in border_corners:
                crn[uv].pin_uv = False

            coords_before = [crn[uv].uv.copy() for crn in border_corners]

            save_transform_islands = []
            for isl in islands:
                save_transform_islands.append(isl.save_transform())

            relax_data.append(RelaxData(umesh, to_select, coords_before, border_corners, save_transform_islands))

        self.relax_b(relax_data)

    def relax_b(self, relax_data):
        # Relax
        bpy.ops.uv.minimize_stretch(iterations=self.iterations)
        if any(rd.coords_before for rd in relax_data):
            bpy.ops.uv.unwrap(method='CONFORMAL')
            # Blend Borders
            for rd in relax_data:
                uv = rd.umesh.uv_layer
                for co, crn in zip(rd.coords_before, rd.border_corners):
                    crn_uv = crn[uv]
                    crn_uv.uv = vec_lerp(co, crn_uv.uv, self.border_blend)
        bpy.ops.uv.minimize_stretch(iterations=self.iterations)

        for rd in relax_data:
            for isl in rd.save_transform_islands:  # TODO: Weld half selected islands
                isl.inplace()

        for rd in relax_data:
            for elem in rd.selected_elem:
                elem.select = False

        for rd in relax_data:
            uv = rd.umesh.uv_layer
            for f in rd.umesh.bm.faces:
                for crn in f.loops:
                    crn[uv].pin_uv = False

    def relax_non_sync(self):
        from ..utils import linked_crn_uv, is_boundary

        relax_data: list[RelaxData] = []
        for umesh in reversed(self.umeshes):
            uv = umesh.uv_layer
            if umesh.is_full_face_deselected or not any(crn[uv].select for f in umesh.bm.faces if f.select for crn in f.loops):
                self.umeshes.umeshes.remove(umesh)
                continue

            islands = Islands.calc_extended_any_elem(umesh.bm, umesh.uv_layer, self.sync)

            for isl in islands:
                isl.mark_seam()

            border_corners = set()

            for f in umesh.bm.faces:
                if not f.select:
                    continue
                for crn in f.loops:
                    crn_uv = crn[uv]
                    if crn_uv.select:
                        if is_boundary(crn, uv):
                            border_corners.add(crn)
                            for crn_ in linked_crn_uv(crn, uv):
                                if crn_.face.select:
                                    border_corners.add(crn_)

            for f in umesh.bm.faces:
                for crn in f.loops:
                    crn[uv].pin_uv = True

            for crn_ in border_corners:
                crn_[uv].pin_uv = False

            coords_before = [crn[uv].uv.copy() for crn in border_corners]

            save_transform_islands = []
            for isl in islands:
                save_transform_islands.append(isl.save_transform())

            relax_data.append(RelaxData(umesh, [], coords_before, border_corners, save_transform_islands))

        self.relax_b(relax_data)
