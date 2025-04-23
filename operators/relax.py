# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy

from . import unwrap
from .. import types
from .. import utils

from ..types import Islands

class RelaxData:
    def __init__(self, _umesh: types.UMesh, _selected_elem, _coords_before, _border_corners, _save_transform_islands):
        self.umesh = _umesh
        self.selected_elem = _selected_elem
        self.coords_before = _coords_before
        self.border_corners = _border_corners
        self.save_transform_islands = _save_transform_islands

    def remove_all_pins_from_umesh(self):
        uv = self.umesh.uv
        for f in self.umesh.bm.faces:
            for crn in f.loops:
                crn[uv].pin_uv = False

class UNIV_OT_Relax(unwrap.UNIV_OT_Unwrap):
    bl_idname = "uv.univ_relax"
    bl_label = "Relax"
    bl_description = "Warning: Incorrect behavior with flipped islands"
    bl_options = {'REGISTER', 'UNDO'}

    iterations: bpy.props.IntProperty(name='Iterations', default=20, min=5, max=150, soft_max=50)
    legacy: bpy.props.BoolProperty(name='Legacy Behavior', default=False)
    border_blend: bpy.props.FloatProperty(name='Border Blend', default=0.1, min=0, soft_min=0, soft_max=1)
    use_correct_aspect: bpy.props.BoolProperty(name='Correct Aspect', default=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        if self.slim_support and not self.legacy and unwrap.MULTIPLAYER != 1:
            self.layout.label(text=f'Multiplayer: x{unwrap.MULTIPLAYER}')
        self.layout.prop(self, 'iterations', slider=True)
        if not self.slim_support or self.legacy:
            self.layout.prop(self, 'border_blend', slider=True)
        if self.slim_support:
            self.layout.prop(self, 'legacy')
        self.layout.prop(self, 'use_correct_aspect')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slim_support: bool = bpy.app.version >= (4, 3, 0)
        if self.slim_support:
            self.unwrap = 'MINIMUM_STRETCH'

    def execute(self, context):

        self.umeshes = types.UMeshes()
        self.umeshes.fix_context()

        # self.umeshes.elem_mode
        # Legacy
        if not self.slim_support or self.legacy:
            self.umeshes.filter_by_selected_uv_elem_by_mode()
            if self.umeshes.sync:  # noqa pycharm moment
                if self.umeshes.elem_mode == 'FACE':
                    self.relax_sync_faces()
                else:
                    self.relax_sync_verts_edges()
            else:
                self.relax_non_sync()

            for umesh in self.umeshes:
                umesh.bm.select_flush_mode()
        else:  # SLIM
            selected_umeshes, unselected_umeshes = self.umeshes.filtered_by_selected_and_visible_uv_verts()
            self.umeshes = selected_umeshes if selected_umeshes else unselected_umeshes
            if not self.umeshes:
                return self.umeshes.update()

            if not selected_umeshes and self.max_distance is not None:
                return self.pick_unwrap(no_flip=True, iterations=self.iterations)

            if self.umeshes.sync:
                if self.umeshes.elem_mode == 'FACE':
                    self.unwrap_sync_faces(no_flip=True, iterations=self.iterations)
                else:
                    self.unwrap_sync_verts_edges(no_flip=True, iterations=self.iterations)
            else:
                self.unwrap_non_sync(no_flip=True, iterations=self.iterations)

        return self.umeshes.update()

    def relax_sync_verts_edges(self):
        relax_data: list[RelaxData] = []
        for umesh in self.umeshes:
            if self.umeshes.elem_mode == 'VERTEX':
                selected_elem = utils.calc_selected_verts(umesh)
            else:
                selected_elem = utils.calc_selected_edges(umesh)

            uv = umesh.uv
            islands = Islands.calc_visible(umesh)
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

            if self.umeshes.elem_mode == 'EDGE':
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

    def relax_a(self, relax_data: list[RelaxData]):
        # Relax
        bpy.ops.uv.minimize_stretch(iterations=self.iterations*5)
        if any(rd.coords_before for rd in relax_data):
            bpy.ops.uv.unwrap(method='CONFORMAL')
            # Blend Borders
            for rd in relax_data:
                uv = rd.umesh.uv
                for co, crn in zip(rd.coords_before, rd.border_corners):
                    crn_uv_co = crn[uv].uv
                    crn_uv_co[:] = co.lerp(crn_uv_co, self.border_blend)
        bpy.ops.uv.minimize_stretch(iterations=self.iterations*5)

        for rd in relax_data:
            for isl in rd.save_transform_islands:  # TODO: Weld half selected islands
                isl.inplace()
        bpy.ops.uv.select_all(action='DESELECT')

        for rd in relax_data:
            for elem in rd.selected_elem:
                elem.select = True

        for rd in relax_data:
            uv = rd.umesh.uv
            for f in rd.umesh.bm.faces:
                for crn in f.loops:
                    crn[uv].pin_uv = False

    def relax_sync_faces(self):
        assert self.umeshes.elem_mode == 'FACE'
        from ..utils import linked_crn_uv_unordered, shared_is_linked

        relax_data: list[RelaxData] = []
        for umesh in self.umeshes:
            uv = umesh.uv
            islands = Islands.calc_extended(umesh)

            for isl in islands:
                isl.mark_seam()

            to_select = set()
            border_corners = set()
            # Find border from selection corners
            for f in utils.calc_selected_uv_faces(umesh):
                for crn in f.loops:
                    linked_crn = linked_crn_uv_unordered(crn, uv)
                    linked_crn.append(crn)
                    border = False
                    for _crn in linked_crn:
                        if _crn.face.hide:
                            continue
                        if (next_linked_disc := _crn.link_loop_radial_prev) == _crn:
                            border = True
                            continue

                        next_face = next_linked_disc.face
                        if next_face.select:
                            continue
                        if next_face.hide:
                            border = True
                            continue

                        if shared_is_linked(next_linked_disc, _crn, uv):
                            to_select.add(next_linked_disc.face)
                        border = True
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

    def relax_b(self, relax_data: list[RelaxData]):
        # Relax
        bpy.ops.uv.minimize_stretch(iterations=self.iterations*5)
        if any(rd.coords_before for rd in relax_data):
            bpy.ops.uv.unwrap(method='CONFORMAL')
            # Blend Borders
            for rd in relax_data:
                uv = rd.umesh.uv
                for co, crn in zip(rd.coords_before, rd.border_corners):
                    crn_uv = crn[uv]
                    crn_uv.uv = co.lerp(crn_uv.uv, self.border_blend)
        bpy.ops.uv.minimize_stretch(iterations=self.iterations*5)

        for rd in relax_data:
            for isl in rd.save_transform_islands:  # TODO: Fix, weld half selected islands
                isl.inplace()

            for elem in rd.selected_elem:
                elem.select = False

            rd.remove_all_pins_from_umesh()

    def relax_non_sync(self):
        from ..utils import linked_crn_uv_unordered, is_boundary_non_sync

        relax_data: list[RelaxData] = []
        for umesh in self.umeshes:
            uv = umesh.uv
            islands = Islands.calc_extended_any_elem(umesh)

            for isl in islands:
                isl.mark_seam()

            border_corners_for_unwrap = set()

            for f in utils.calc_selected_uv_faces(umesh):
                for crn in f.loops:
                    crn_uv = crn[uv]
                    if crn_uv.select:
                        if is_boundary_non_sync(crn, uv):
                            border_corners_for_unwrap.add(crn)
                            for crn_ in linked_crn_uv_unordered(crn, uv):
                                if crn_.face.select:
                                    border_corners_for_unwrap.add(crn_)

            for f in umesh.bm.faces:
                for crn in f.loops:
                    crn[uv].pin_uv = True

            for crn_ in border_corners_for_unwrap:
                crn_[uv].pin_uv = False

            coords_before = [crn[uv].uv.copy() for crn in border_corners_for_unwrap]

            save_transform_islands = []
            for isl in islands:
                save_transform_islands.append(isl.save_transform())

            relax_data.append(RelaxData(umesh, [], coords_before, border_corners_for_unwrap, save_transform_islands))

        self.relax_b(relax_data)
