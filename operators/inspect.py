# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import math
import enum
import bl_math
import textwrap

from mathutils.geometry import area_tri
from bpy.props import *
from bpy.types import Operator

from .. import utils
from .. import types

class Inspect(enum.IntFlag):
    Overlap = enum.auto()
    OverlapInexact = enum.auto()
    OverlapSelf = enum.auto()
    OverlapTroubleFace = enum.auto()
    OverlapByMaterial = enum.auto()
    OverlapWithModifier = enum.auto()
    __pass1 = enum.auto()
    __pass2 = enum.auto()

    AllOverlapFlags = (Overlap | OverlapWithModifier | OverlapInexact |
                       OverlapSelf | OverlapTroubleFace | OverlapByMaterial)
    Zero = enum.auto()
    __pass3 = enum.auto()

    OverlapByPadding = enum.auto()
    DoubleVertices3D = enum.auto()
    TileIsect = enum.auto()


    Flipped = enum.auto()
    Flipped3D = enum.auto()
    InsideNormalsOrient = enum.auto()
    Rotated = enum.auto()
    __pass4 = enum.auto()

    NonManifold = enum.auto()
    NonSplitted = enum.auto()

    OverScaled = enum.auto()
    OverStretched = enum.auto()
    AngleStretch = enum.auto()
    __pass5 = enum.auto()

    Concave = enum.auto()
    DeduplicateUVLayers = enum.auto()
    RepairAfterJoin = enum.auto()
    __pass6 = enum.auto()
    IncorrectBMeshTags = enum.auto()
    Other = enum.auto()

    @classmethod
    def default_value_for_settings(cls):
        return cls.Overlap | cls.Zero | cls.Flipped | cls.NonSplitted | cls.Other

class UNIV_OT_Check_Zero(Operator):
    bl_idname = "uv.univ_check_zero"
    bl_label = "Zero"
    bl_description = "Select degenerate UVs (zero area UV triangles)"
    bl_options = {'REGISTER', 'UNDO'}

    precision: FloatProperty(name='Precision', default=1e-6, min=0, soft_max=0.001, step=0.0001, precision=7)  # noqa

    def draw(self, context):
        self.layout.prop(self, 'precision', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = types.UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False
        bpy.ops.uv.select_all(action='DESELECT')

        total_counter = self.zero(umeshes, self.precision)
        umeshes.update()

        if not total_counter:
            self.report({'INFO'}, 'Degenerate triangles not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} degenerate triangles')
        return {'FINISHED'}

    @staticmethod
    def zero(umeshes, precision=1e-6):
        sync = umeshes.sync
        if sync and umeshes.elem_mode != 'FACE':
            umeshes.elem_mode = 'FACE'

        precision *= 0.001
        total_counter = 0
        select_set = utils.face_select_linked_func(sync)
        for umesh in umeshes:
            if not sync and umesh.is_full_face_deselected:
                continue

            uv = umesh.uv
            local_counter = 0
            for tris_a, tris_b, tris_c in umesh.bm.calc_loop_triangles():
                face = tris_a.face
                if face.hide if sync else not face.select:
                    continue

                area = area_tri(tris_a[uv].uv, tris_b[uv].uv, tris_c[uv].uv)
                if area <= precision:
                    select_set(face, uv)
                    local_counter += 1
            umesh.update_tag |= bool(local_counter)
            total_counter += local_counter
        return total_counter


class UNIV_OT_Check_Flipped(Operator):
    bl_idname = "uv.univ_check_flipped"
    bl_label = "Flipped"
    bl_description = "Select flipped UV faces"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = types.UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False
        bpy.ops.uv.select_all(action='DESELECT')

        total_counter = self.flipped(umeshes)
        umeshes.update()

        if not total_counter:
            self.report({'INFO'}, 'Flipped faces not found')
            return {'FINISHED'}

        self.report({'WARNING'}, f'Detected {total_counter} flipped faces')
        return {'FINISHED'}

    @staticmethod
    def flipped(umeshes):
        sync = umeshes.sync
        if sync and umeshes.elem_mode != 'FACE':
            umeshes.elem_mode = 'FACE'

        total_counter = 0
        select_set = utils.face_select_linked_func(sync)
        for umesh in umeshes:
            if not sync and umesh.is_full_face_deselected:
                continue

            uv = umesh.uv
            local_counter = 0
            for tris_a, tris_b, tris_c in umesh.bm.calc_loop_triangles():
                face = tris_a.face
                if face.hide if sync else not face.select:
                    continue

                a = tris_a[uv].uv
                b = tris_b[uv].uv
                c = tris_c[uv].uv

                signed_area = a.cross(b) + b.cross(c) + c.cross(a)
                if signed_area < 0:
                    select_set(face, uv)
                    local_counter += 1
            umesh.update_tag |= bool(local_counter)
            total_counter += local_counter
        return total_counter


class UNIV_OT_Check_Non_Splitted(Operator):
    bl_idname = 'uv.univ_check_non_splitted'
    bl_label = 'Non-Splitted'
    bl_description = "Selects the edges where seams should be marked and unwrapped without connection"
    bl_options = {'REGISTER', 'UNDO'}

    use_auto_smooth: bpy.props.BoolProperty(name='Use Auto Smooth', default=True)
    user_angle: FloatProperty(name='Smooth Angle', default=math.radians(66.0), subtype='ANGLE', min=math.radians(5.0), max=math.radians(180.0))

    def draw(self, context):
        layout = self.layout
        layout.prop(self, 'use_auto_smooth')
        layout.prop(self, 'user_angle', slider=True)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def execute(self, context):
        umeshes = types.UMeshes()
        umeshes.fix_context()
        bpy.ops.uv.select_all(action='DESELECT')

        # clamp angle
        if self.use_auto_smooth:
            max_angle_from_obj_smooth = max(umesh.smooth_angle for umesh in umeshes)
            self.user_angle = bl_math.clamp(self.user_angle, 0.0, max_angle_from_obj_smooth)

        result = self.non_splitted(umeshes, self.use_auto_smooth, self.user_angle)
        if formatted_text := self.data_formatting(result):
            self.report({'WARNING'}, formatted_text)
            umeshes.update()
        else:
            self.report({'INFO'}, 'All visible edges are splitted.')

        return {'FINISHED'}

    @staticmethod
    def non_splitted(umeshes, use_auto_smooth, user_angle, batch_inspect=False):
        non_seam_counter = 0
        non_manifold_counter = 0
        angle_counter = 0
        sharps_counter = 0
        seam_counter = 0
        mtl_counter = 0

        sync = umeshes.sync
        is_boundary = utils.is_boundary_sync if sync else utils.is_boundary_non_sync

        for umesh in umeshes:
            to_select = set()
            if use_auto_smooth:
                angle = min(umesh.smooth_angle, user_angle)
            else:
                angle = user_angle

            uv = umesh.uv
            for crn in utils.calc_visible_uv_corners(umesh):
                if (pair_crn := crn.link_loop_radial_prev) in to_select:
                    continue

                edge = crn.edge
                if is_boundary(crn, uv):
                    if crn.edge.seam:
                        continue
                    if edge.is_boundary:
                        continue
                    non_seam_counter += 1
                elif not edge.smooth:
                    sharps_counter += 1
                elif (face_angle := edge.calc_face_angle(1000.0)) >= angle:
                    if face_angle == 1000.0:
                        non_manifold_counter += 1  # TODO: Check if batch_inspect disabled, after implement non manifold
                    else:
                        angle_counter += 1
                elif edge.seam:
                    seam_counter += 1
                elif pair_crn.face.material_index != crn.face.material_index:
                    mtl_counter += 1
                else:
                    continue
                to_select.add(crn)

            umesh.sequence = to_select
            umesh.update_tag |= bool(to_select)

        if any(umesh.sequence for umesh in umeshes):
            if batch_inspect:
                # Avoid switch elem mode if all edges selected for batch inspect
                select_get = utils.edge_select_linked_get_func(sync)
                all_selected = True
                for umesh in umeshes:
                    uv = umesh.uv
                    if not all(select_get(crn_edge, uv) for crn_edge in umesh.sequence):
                        all_selected = False
                        break
                if all_selected:
                    for umesh in umeshes:
                        umesh.sequence = []
                    return non_seam_counter, non_manifold_counter, angle_counter, sharps_counter, seam_counter, mtl_counter

            select_set = utils.edge_select_linked_set_func(sync)
            if umeshes.elem_mode not in ('EDGE', 'VERT'):
                umeshes.elem_mode = 'EDGE'

            for umesh in umeshes:
                uv = umesh.uv
                for edge in umesh.sequence:
                    select_set(edge, True, uv)
                umesh.sequence = []
        return non_seam_counter, non_manifold_counter, angle_counter, sharps_counter, seam_counter, mtl_counter

    @staticmethod
    def data_formatting(counters):
        non_seam_counter, non_manifold_counter, angle_counter, sharps_counter, seam_counter, mtl_counter = counters
        r_text = ''
        if non_seam_counter:
            r_text += f'Non-Seam - {non_seam_counter}. '
        if non_manifold_counter:
            r_text += f'Non-Manifolds - {non_manifold_counter}. '
        if angle_counter:
            r_text += f'Sharp Angles - {angle_counter}. '
        if sharps_counter:
            r_text += f'Mark Sharps - {sharps_counter}. '
        if seam_counter:
            r_text += f'Mark Seams - {seam_counter}. '
        if mtl_counter:
            r_text += f'Materials - {mtl_counter}. '

        if r_text:
            r_text = f'Found: {sum(counters)} non splitted edges. {r_text}'
        return r_text


class UNIV_OT_Check_Overlap(Operator):
    bl_idname = 'uv.univ_check_overlap'
    bl_label = 'Overlap'
    bl_description = "Select all UV faces which overlap each other.\n" \
                     "Unlike the default operator, this one informs about the number of faces with conflicts"
    bl_options = {'REGISTER', 'UNDO'}

    check_mode: EnumProperty(name='Check Overlaps Mode', default='ALL', items=(('ALL', 'All', ''), ('INEXACT', 'Inexact', '')))
    threshold: bpy.props.FloatProperty(name='Distance', default=0.0008, min=0.0, soft_min=0.00005, soft_max=0.00999)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        if self.check_mode == 'INEXACT':
            layout.prop(self, 'threshold')
        layout.row(align=True).prop(self, 'check_mode', expand=True)

    def execute(self, context):
        umeshes = types.UMeshes()
        umeshes.fix_context()
        umeshes.update_tag = False

        count = self.overlap_check(umeshes, self.check_mode, self.threshold)
        self.report(*self.get_info_from_count(count, self.check_mode))
        umeshes.silent_update()
        return {'FINISHED'}

    @staticmethod
    def get_info_from_count(count, check_mode):
        if count:
            if check_mode == 'INEXACT':
                return {'WARNING'}, f"Found {count} islands with inexact overlap"
            else:
                return {'WARNING'}, f"Found about {count} edges with overlap"
        else:
            if check_mode == 'INEXACT':
                return {'INFO'}, f"Inexact islands with overlap not found"
            else:
                return {'INFO'}, f"Edges with overlap not found"

    @staticmethod
    def overlap_check(umeshes, check_mode, threshold=0.001) -> int:
        if umeshes.sync and umeshes.elem_mode != 'FACE':
            umeshes.elem_mode = 'FACE'

        bpy.ops.uv.select_overlap()

        count = 0
        if check_mode == 'INEXACT':
            all_islands = []
            for umesh in umeshes:
                adv_islands = types.AdvIslands.calc_extended_with_mark_seam(umesh)
                # The following subdivision is needed to ignore the exact self overlaps that are created from the flipped face
                for isl in reversed(adv_islands):
                    if isl.has_flip_with_noflip():
                        adv_islands.islands.remove(isl)
                        noflip, flipped = isl.calc_islands_by_flip_with_mark_seam()
                        adv_islands.islands.extend(noflip)
                        adv_islands.islands.extend(flipped)
                all_islands.extend(adv_islands)

            overlapped = types.UnionIslands.calc_overlapped_island_groups(all_islands, threshold)
            for isl in overlapped:
                if isinstance(isl, types.AdvIsland):
                    count += 1
                else:
                    isl.select = False
                    isl.umesh.update_tag = True
        else:
            for umesh in umeshes:
                if umesh.sync:
                    count += umesh.total_edge_sel
                else:
                    count += len(utils.calc_selected_uv_edge_corners(umesh))

        return count


class UNIV_OT_Check_Other(Operator):
    bl_idname = 'uv.univ_check_other'
    bl_label = 'Other'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = """This operator includes a number of small but important checks:\n
1. Verifies that the object contains polygons.
2. Verifies the presence of UV maps.
3. Corrects the operator call context (since many operators fail when the active object lacks polygons or UV maps).
4. Checks for applied or non-uniform object scaling.
5. Ensures a valid aspect ratio.
6. Verifies that the UV Map node in the shader has a valid name.
7. Checks whether the optimal UV smoothing method is selected in the Subdivision Surface modifier."""
# 8. Checks handlers and various flags"""

    check_mode: EnumProperty(name='Check Overlaps Mode', default='ALL', items=(('ALL', 'All', ''), ('INEXACT', 'Inexact', '')))
    threshold: bpy.props.FloatProperty(name='Distance', default=0.0008, min=0.0, soft_min=0.00005, soft_max=0.00999)

    @classmethod
    def poll(cls, context):
        return (obj := context.active_object) and obj.type == 'MESH'

    def draw(self, context):

        layout = self.layout
        col = layout.column()
        global INSPECT_INFO

        if info_list := INSPECT_INFO.get('Other'):
            for check_type, info in info_list:
                box = col.box()
                wrapped_lines = textwrap.wrap(check_type+': '+info, width=72)
                for line in wrapped_lines:
                    box.label(text=line)

    def execute(self, context):
        global INSPECT_INFO
        INSPECT_INFO.clear()

        umeshes = types.UMeshes()
        umeshes.update_tag = False

        if info := self.check_other(umeshes):
            INSPECT_INFO['Other'] = info
            # umeshes.silent_update()
            return context.window_manager.invoke_popup(self, width=420)
        else:
            self.report({'INFO'}, 'Errors not found')
        return {'FINISHED'}

    @staticmethod
    def check_other(umeshes: types.UMeshes):
        info: list[tuple[str, str]] = []

        # Check context error
        faces = bpy.context.active_object.data.polygons
        uvs = bpy.context.active_object.data.uv_layers
        if not faces or not uvs:
            umeshes.fix_context()
            error_description = 'The active object has '
            if not faces:
                error_description += 'no polygons'
            if not uvs:
                if not faces:
                    error_description += ' and '
                error_description += 'no UV maps'

            error_id = 'Context Error'
            if not bpy.context.active_object.data.polygons or not bpy.context.active_object.data.uv_layers:
                info.append((error_id, error_description))
                return info
            else:
                error_description += '. (Fixed)'
                info.append((error_id, error_description))

        selected_objects = utils.calc_any_unique_obj()

        # Check unapplied scale
        counter = 0
        error_id = 'Unapplied Scales'
        error_description = ''
        for obj in selected_objects:
            _, _, scale = obj.matrix_world.decompose()
            if not utils.umath.vec_isclose_to_uniform(scale, abs_tol=0.01):
                counter += 1
                if counter == 1:
                    error_description = f"The {obj.name!r} hasn't applied scale: X={scale.x:.4f}, Y={scale.y:.4f}, Z={scale.z:.4f}"
        if counter:
            if counter != 1:
                error_description = f'Found {counter} objects without applied scales.'
            info.append((error_id, error_description))

        # Check UV maps exist
        uv_names: set[tuple[str, ...]] = set()
        uv_active: set[tuple[bool, ...]] = set()
        uv_active_render: set[tuple[bool, ...]] = set()
        min_uv_maps = 100
        max_uv_maps = 0
        missed_uvs_counter = 0

        error_id = 'UV Maps'
        for obj in selected_objects:
            uvs = obj.data.uv_layers
            if not len(uvs):
                missed_uvs_counter += 1  # TODO: HP check
                continue
            max_uv_maps = max(max_uv_maps, len(uvs))
            min_uv_maps = min(min_uv_maps, len(uvs))
            uv_names.add(tuple(uv.name for uv in uvs))
            uv_active.add(tuple(uv.active for uv in uvs))
            uv_active_render.add(tuple(uv.active_render for uv in uvs))

        error_description = ''
        if missed_uvs_counter:
            error_description += f'Found {missed_uvs_counter} meshes with missed UV maps. '
        if min_uv_maps != max_uv_maps:
            error_description += 'Meshes have different numbers of UVs'
        else:
            if len(uv_names) >= 2:
                error_description += 'Meshes have different names.'
            if len(uv_active) >= 2 or len(uv_active_render) >= 2:
                error_description += 'Meshes have different '
                if len(uv_active) >= 2:
                    error_description += 'active layers'
                if len(uv_active_render) >= 2:
                    if len(uv_active) >= 2:
                        error_description += 'and different active render layers'
                    else:
                        error_description += 'active render layers'
        if error_description:
            info.append((error_id, error_description))

        # Check missed faces
        counter = 0
        error_id = 'Missed Faces'
        error_description = ''
        for obj in selected_objects:
            if not obj.data.polygons:
                counter += 1
                if counter == 1:
                    error_description += f"{obj.name!r}"
                else:
                    error_description += f", {obj.name!r}"

        if counter:
            info.append((error_id, f"{error_description} meshes hasn't faces."))

        if umeshes.is_edit_mode:
            counter = 0
            mode = utils.get_select_mode_mesh()
            for umesh in umeshes:
                if mode not in umesh.bm.select_mode:
                    counter += 1
                    umesh.bm.select_mode = {mode}
            if counter:
                info.append(('Elem Mode', f"Select Mode at {counter} meshes doesn't match scene with mesh. (Fixed)"))


        # Check Heterogeneous Aspect Ratio
        aspects = {}
        def get_aspects_ratio(u):
            if modifiers := [m for m in u.obj.modifiers if m.name.startswith('UniV Checker')]:
                socket = 'Socket_1' if 'Socket_1' in modifiers[0] else 'Input_1'
                if mtl := modifiers[0][socket]:
                    for node in mtl.node_tree.nodes:
                        if node.bl_idname == 'ShaderNodeTexImage' and (image := node.image):
                            image_width, image_height = image.size
                            if image_height:
                                aspects[image_width / image_height] = modifiers[0][socket].name
                            else:
                                aspects[1.0] = mtl.name
                            return
                aspects[1.0] = 'Default'

            # Aspect from material
            elif u.obj.material_slots:
                for slot in u.obj.material_slots:
                    mtl = slot.material
                    if not slot.material:
                        aspects[1.0] = 'Default'
                        continue
                    if mtl.use_nodes and (active_node := mtl.node_tree.nodes.active):
                        if active_node.bl_idname == 'ShaderNodeTexImage' and (image := active_node.image):
                            image_width, image_height = image.size
                            if image_height:
                                aspects[image_width / image_height] = mtl.name
                                continue
                    aspects[1.0] = mtl.name
            else:
                aspects[1.0] = 'Default'

        for umesh in umeshes:
            get_aspects_ratio(umesh)
        if len(aspects) >= 2:
            info.append(('Aspect Ratio', f'Found heterogeneous aspect ratio: ' + ', '.join(f"{v!r}" for v in aspects.values())))

        # Check non valid UV Map node
        error_description = ''
        for umesh in umeshes:
            uv_maps_names = {uv.name for uv in umesh.obj.data.uv_layers}
            uv_maps_names.add('')
            for slot in umesh.obj.material_slots:
                if (mtl := slot.material) and mtl.use_nodes:
                    if non_valid_names := {f'{node.uv_map!r}' for node in mtl.node_tree.nodes
                                       if node.type == 'UVMAP' and node.uv_map not in uv_maps_names}:
                        error_description += f'Material {mtl.name!r} in {umesh.obj.name!r} object has non-valid UV Map name: '
                        error_description += ', '.join(non_valid_names) + '.\n'
        if error_description:
            info.append(('UV Map None Name', error_description))

        # Check SubDiv optimal method for UV
        names = []
        for u in umeshes:
            for mod in u.obj.modifiers:
                if isinstance(mod, bpy.types.SubsurfModifier):
                    if mod.uv_smooth in ('PRESERVE_BOUNDARIES', 'NONE'):
                        names.append(repr(u.obj.name))
                        break
        if names:
            error_description = 'The following objects use a non-optimal UV smoothing method in subdivision modifier: ' + ', '.join(names)
            error_description += '. Use Keep Corners or All Method'
            info.append(('UV Smooth', error_description))

        return info



class UNIV_OT_BatchInspectFlags(Operator):
    bl_idname = 'uv.univ_batch_inspect_flags'
    bl_label = 'Flags'
    bl_description = "Inspect Flags"

    flag: IntProperty(name='Flag', default=0, min=0)

    def execute(self, context):
        from ..preferences import univ_settings
        tag = Inspect(self.flag)

        # Turn off the old (all) overlap flag if it doesn't match the new one.
        if tag in Inspect.AllOverlapFlags:
            if tag not in Inspect(univ_settings().batch_inspect_flags):
                univ_settings().batch_inspect_flags &= ~Inspect.AllOverlapFlags

        univ_settings().batch_inspect_flags ^= tag
        return {'FINISHED'}

INSPECT_INFO = {}

class UNIV_OT_BatchInspect(Operator):
    bl_idname = 'uv.univ_batch_inspect'
    bl_label = 'Inspect'
    bl_description = "Batch Inspect by Enabled tags"
    bl_options = {'REGISTER', 'UNDO'}

    inspect_all: BoolProperty(name='Inspect All', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        layout = self.layout
        col = layout.column()
        if 'Hidden' in INSPECT_INFO:
            box = col.box()
            box.label(text='Hidden faces found — the result may be incorrect', icon='INFO')
            col.separator()

        for inspect_flag in ('Overlap', 'Zero', 'Flipped', 'Non-Splitted'):
            if info := INSPECT_INFO.get(inspect_flag):
                box = col.box()
                box.label(text=f'{inspect_flag}: ' + info)

        if info_list := INSPECT_INFO.get('Other'):
            for check_type, info in info_list:
                box = col.box()
                wrapped_lines = textwrap.wrap(check_type+': '+info, width=72)
                for line in wrapped_lines:
                    box.label(text=line)

    def invoke(self, context, event):
        if event.value == 'PRESS':
            return self.execute(context)

        self.inspect_all = event.alt
        return self.execute(context)

    def execute(self, context):
        from ..preferences import univ_settings
        flags = Inspect(univ_settings().batch_inspect_flags)
        global INSPECT_INFO
        INSPECT_INFO.clear()
        if not flags:
            self.report({'WARNING'}, 'Not found enabled flags')
            return {'CANCELLED'}

        umeshes = types.UMeshes()
        umeshes.update_tag = False

        if (Inspect.Other in flags) or self.inspect_all:
            if info := UNIV_OT_Check_Other.check_other(umeshes):
                INSPECT_INFO['Other'] = info
                if not umeshes:
                    return context.window_manager.invoke_popup(self, width=420)

        if not umeshes:
            has_uv_maps = any(obj.data.uv_layers for obj in bpy.context.selected_objects if obj.type == 'MESH')
            self.report({'WARNING'}, 'Not found meshes with polygons' if has_uv_maps else 'Not found meshes with UV maps')
            return {'CANCELLED'}

        if (flags & Inspect.AllOverlapFlags) or self.inspect_all:
            if Inspect.Overlap in flags or (self.inspect_all and not (flags & Inspect.AllOverlapFlags)):
                if count := UNIV_OT_Check_Overlap.overlap_check(umeshes, 'ALL'):
                    INSPECT_INFO['Overlap'] = UNIV_OT_Check_Overlap.get_info_from_count(count, 'ALL')[1]

            elif Inspect.OverlapInexact in flags:
                if count := UNIV_OT_Check_Overlap.overlap_check(umeshes, 'INEXACT'):
                    INSPECT_INFO['Overlap'] = UNIV_OT_Check_Overlap.get_info_from_count(count, 'INEXACT')[1]
        else:
            bpy.ops.uv.select_all(action='DESELECT')

        if Inspect.Zero in flags or self.inspect_all:
            if count := UNIV_OT_Check_Zero.zero(umeshes):
                INSPECT_INFO['Zero'] = f'Detected {count} degenerate triangles'

        if Inspect.Flipped in flags or self.inspect_all:
            if count := UNIV_OT_Check_Flipped.flipped(umeshes):
                INSPECT_INFO['Flipped'] = f'Detected {count} flipped faces'

        if Inspect.NonSplitted in flags or self.inspect_all:  # Last check, because it switches elem mode to EDGE.
            result = UNIV_OT_Check_Non_Splitted.non_splitted(umeshes, use_auto_smooth=True, user_angle=180, batch_inspect=True)
            if info := UNIV_OT_Check_Non_Splitted.data_formatting(result):
                INSPECT_INFO['Non-Splitted'] = info


        has_hidden_faces = False
        for umesh in umeshes:
            if has_hidden_faces:
                break
            if umesh.is_full_face_selected:
                continue

            if umeshes.sync:
                for f in umesh.bm.faces:
                    if f.hide:
                        has_hidden_faces = True
                        break
            else:
                for f in umesh.bm.faces:
                    if not f.select:
                        has_hidden_faces = True
                        break
        if has_hidden_faces:
            if not INSPECT_INFO:
                self.report({'WARNING'}, 'No problems detected, but hidden faces could lead to incorrect results.')
                return {'FINISHED'}
            INSPECT_INFO['Hidden'] = True

        if not INSPECT_INFO:
            self.report({'INFO'}, 'No errors detected.')
        else:
            umeshes.silent_update()
            return context.window_manager.invoke_popup(self, width=420)
        return {'FINISHED'}
