# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import typing
import numpy as np
import numpy.typing as npt

from .. import utils
from ..utils import linked_crn_to_vert_by_idx_without_co_check_unordered
from .. import utypes
from ..utypes import UMeshes, AdvIsland, AdvIslands
from collections import deque, defaultdict
from collections.abc import Callable

from bmesh.types import BMFace, BMLoop

# TODO: Rename target to reference
T = typing.TypeVar('T')


class FacePattern:
    def __init__(self, f: BMFace, start_crn: BMLoop, ordered_corners):
        self.face: BMFace = f
        self.start_crn: BMLoop = start_crn
        self.ordered_corners: deque[BMLoop] = ordered_corners  # Face ordered corners by start corner (included and not included)
        # TODO: Add more comments about this
        self.ordered_corners_pair_crn_face_sides: npt.NDArray[np.uint16] = np.zeros(shape=len(ordered_corners), dtype='uint16')

    @classmethod
    def calc_init(cls, f, start_crn):
        return cls(f, start_crn, deque(f.loops))

    @classmethod
    def calc_fw(cls, f: BMFace, start_crn: BMLoop):
        linked = deque(f.loops)
        linked.rotate(-linked.index(start_crn))
        linked.popleft()
        return cls(f, start_crn, linked)

    @classmethod
    def calc_bw(cls, f, start_crn):
        linked = deque(reversed(f.loops))
        linked.rotate(-linked.index(start_crn))
        linked.popleft()
        return cls(f, start_crn, linked)

    def __str__(self):
        return f"FacePattern: shared crn face size = {self.ordered_corners_pair_crn_face_sides}"

    def __iter__(self) -> typing.Iterator[BMLoop]:
        return iter(self.ordered_corners)

    def __setitem__(self, key: int, value: int):
        self.ordered_corners_pair_crn_face_sides[key] = value


class StackIsland:  # TODO: Split for target and target islands
    def __init__(self, island):
        self.island: AdvIsland = island
        self.walked_island_from_init_face: list[list[FacePattern]] = []

        self.unique_faces: list[BMFace] = []

        # Forward patterns.
        self.face_start_pattern_fw: deque[list[int]] = deque()  # Linked sorted face sides count.
        self.face_start_pattern_crn_fw: deque[BMLoop] = deque()  # Ordered by unique corner - corners.
        self.start_faces_patterns_fw: list[list[deque[list[int]] | deque[BMLoop]]] = []  # Lazy computed patterns (taken by `calc_unique_faces`)

        # Backward patterns.
        self.face_start_pattern_bw: deque[list[int]] = deque()
        self.face_start_pattern_crn_bw: deque[BMLoop] = deque()
        self.start_faces_patterns_bw: list[list[deque[list[int]] | deque[BMLoop]]] = []

        self.poligon_sides_with_faces: defaultdict[int, list[BMFace]] = defaultdict(list)

        # Used for quantity stack groups.
        self.np_ngons_with_faces: npt.NDArray[np.int32] = np.array([], dtype='int32')

    def preprocessing(self):
        self.ensure_polygon_sides_with_faces()
        # Get more unique faces for start walking, for minimize iterations
        self.calc_unique_faces()
        # Get the largest face
        self.unique_faces.sort(key=lambda face: face.calc_area(), reverse=True)
        # Need for grouping by same polygons sizes
        self.ngons_to_np()
        # Get first pattern for walking. Once the first pattern passes, the other patterns are processed immediately.
        self.face_start_pattern_fw, self.face_start_pattern_crn_fw = self.calc_linked_corners_pattern(idx=0)

    def ensure_polygon_sides_with_faces(self):
        for f in self.island:
            self.poligon_sides_with_faces[len(f.loops)].append(f)

    def calc_unique_faces(self) -> list[BMFace]:
        assert self.poligon_sides_with_faces

        min_sequence: tuple[int | list[BMFace]] | tuple = ()
        for ngons_size, faces in self.poligon_sides_with_faces.items():
            if not min_sequence:
                # Set first unique faces
                min_sequence = ngons_size, faces

            # Set by min faces length
            elif len(faces) < len(min_sequence[1]):
                min_sequence = ngons_size, faces
            # Set by more poligon sides, if faces size equal
            elif len(faces) == len(min_sequence[1]) and ngons_size > min_sequence[0]:
                min_sequence = ngons_size, faces

        self.unique_faces = min_sequence[1]
        return self.unique_faces

    def ngons_to_np(self):
        np_ngons_with_faces = np.empty(shape=(len(self.poligon_sides_with_faces), 2), dtype='int32')
        for idx, (f_size, faces) in enumerate(self.poligon_sides_with_faces.items()):
            np_ngons_with_faces[idx][0] = f_size
            np_ngons_with_faces[idx][1] = len(faces)

        self.np_ngons_with_faces = np_ngons_with_faces[np_ngons_with_faces[:, 0].argsort()]

    @staticmethod
    def get_unique_start_crn_idx(face_start_pattern_crn: deque[BMLoop]) -> int:
        eps = 1e-6
        edge_lengths = np.array([crn.edge.calc_length() for crn in face_start_pattern_crn])

        distance_matrix = np.abs(edge_lengths[:, None] - edge_lengths)
        mean_dist = distance_matrix.mean(axis=1)

        duplicates_count = (distance_matrix < eps).sum(axis=1)

        # TODO: if 1 not in duplicates_count and case all_eq(edge_lengths)- calculate by BBox3D
        #  In the first case, need rotate the array, and if there’s no repeating pattern in the lengths,
        #  then everything is fine; otherwise all_eq(edge_lengths), need calculate the starting index using the BBox3D.

        # TODO: Add option World Align or Local start point
        all_eq = np.allclose(edge_lengths, edge_lengths[0], rtol=0, atol=eps)
        if all_eq:
            from ..utypes import bbox
            vert_3d_coords = [crn.vert.co for crn in face_start_pattern_crn]
            bbox3d_min = bbox.BBox3D.calc_bbox(vert_3d_coords).min
            min_idx = 0
            min_dist = float('inf')
            for idx, co in enumerate(vert_3d_coords):
                if (dist := (bbox3d_min - co).length) < min_dist:
                    min_dist = dist
                    min_idx = idx
            return min_idx

        mean_dist[duplicates_count > 1] *= 0.7  # Apply 30% penalty for elements with duplicates

        unique_dist_idx = np.argmax(mean_dist)
        return unique_dist_idx

    def calc_linked_corners_pattern(self, idx) -> list[deque[list[int]] | deque[BMLoop]]:
        init_face: BMFace = self.unique_faces[idx]
        face_idx = init_face.index

        face_start_pattern: deque[list[int]] = deque()  # [shared face size, linked face size]
        face_start_pattern_crn: deque[BMLoop] = deque(init_face.loops)

        unique_dist_idx = self.get_unique_start_crn_idx(face_start_pattern_crn)
        face_start_pattern_crn.rotate(-unique_dist_idx)

        for crn in face_start_pattern_crn:
            # TODO: Implement radial linked corners by index, with missing not equal index faces, without reset searching.
            linked_crn_faces_sides = [len(l_crn.face.loops) for l_crn in linked_crn_to_vert_by_idx_without_co_check_unordered(crn)]

            shared_face_size = 0
            shared_crn = crn.link_loop_radial_prev
            if shared_crn != crn and shared_crn.face.index == face_idx:
                shared_face_size = len(shared_crn.face.loops)

            linked_crn_faces_sides.sort()
            linked_crn_faces_sides.insert(0, shared_face_size)
            face_start_pattern.append(linked_crn_faces_sides)

        return [face_start_pattern, face_start_pattern_crn]

    def calc_all_linked_corners_pattern_fw(self):
        start_faces_patterns_fw: list[list[deque[list[int]] | deque[BMLoop]]] = []
        for idx, sequ in enumerate(self.unique_faces):
            # Skip first (unique) calculated pattern
            if not idx:
                continue
            start_faces_patterns_fw.append(self.calc_linked_corners_pattern(idx))
        self.start_faces_patterns_fw = start_faces_patterns_fw

    def compared_matching_first_pattern(self, other: 'StackIsland'):
        if len(self.face_start_pattern_fw) != len(other.face_start_pattern_fw):
            return

        # Rotate the start patterns based on their correspondence via linked face sides.
        for _ in range(len(other.face_start_pattern_fw) - 1):
            if self.face_start_pattern_fw == other.face_start_pattern_fw:
                yield other.face_start_pattern_crn_fw
            other.face_start_pattern_fw.rotate(1)
            other.face_start_pattern_crn_fw.rotate(1)


        # Rotate other patterns
        if not other.start_faces_patterns_fw:
            other.calc_all_linked_corners_pattern_fw()

        for face_start_pattern, face_start_pattern_crn in other.start_faces_patterns_fw:
            for _ in range(len(face_start_pattern) - 1):
                if self.face_start_pattern_fw == face_start_pattern:
                    yield face_start_pattern_crn

                face_start_pattern.rotate(1)
                face_start_pattern_crn.rotate(1)


    def calc_transfer_stack_island_fw(self, transfer: 'StackIsland'):
        for transfer_pattern in self.compared_matching_first_pattern(transfer):  # Iterate across all unique faces
            # Container collector of island transfer patterns by generation step
            transfer_island_walked: list[list[FacePattern]] = []
            transfer.island.set_tag(True)  # TODO: Tagging transfer_island_walked

            start_crn = transfer_pattern[0]
            init_face_pattern = FacePattern(start_crn.face, start_crn, transfer_pattern)
            init_face_pattern.face.tag = False

            face_idx = init_face_pattern.face.index

            current_transfer_faces: list[FacePattern] = [init_face_pattern]
            next_transfer_faces = []

            failed = False
            for generation_step_of_shared_face in self.walked_island_from_init_face:
                for target_face, transfer_face in zip(generation_step_of_shared_face, current_transfer_faces):
                    if len(target_face.ordered_corners) != len(transfer_face.ordered_corners):
                        failed = True
                        break

                    for tar_face_sides, transfer_crn in zip(target_face.ordered_corners_pair_crn_face_sides, transfer_face.ordered_corners):
                        shared_crn = transfer_crn.link_loop_radial_prev
                        shared_crn_face = shared_crn.face

                        # Skip boundary edges.
                        if tar_face_sides == 0:
                            if shared_crn == transfer_crn or (not shared_crn_face.tag) or (shared_crn_face.index != face_idx):
                                continue
                            else:
                                # Transfer corner has valid pair corner, stop the walking.
                                failed = True
                                break

                        # Expect pair crn, break.
                        if shared_crn == transfer_crn:
                            failed = True
                            break

                        if tar_face_sides != len(shared_crn_face.loops):
                            failed = True
                            break

                        # Just to be safe, we check the tag along with the index, since the tag may be corrupted on non-manifold edges.
                        if (not shared_crn_face.tag) or shared_crn_face.index != face_idx:
                            failed = True
                            break

                        # TODO: Restore tags after failed from contained faces.
                        shared_crn_face.tag = False
                        next_transfer_faces.append(FacePattern.calc_fw(shared_crn_face, shared_crn))
                    if failed:
                        break

                transfer_island_walked.append(current_transfer_faces)
                current_transfer_faces = next_transfer_faces
                next_transfer_faces = []
                if failed:
                    # NOTE: Only one failing pattern is stopped, while iteration continues over all unique faces.
                    break
            if not failed:
                return transfer_island_walked
        return None

    @staticmethod
    def pattern_to_str(pattern):
        text = ''
        for idx, p in enumerate(pattern):
            text += f'{idx}: {list(p)}, '
        return text

    def calc_walked_reference_island_fw_and_for_bw(self):
        """Create an island that saves a sequence of found faces and its shared corner"""
        self.island.set_tag(True)

        init_face = self.unique_faces[0]
        # For the future be careful, maybe face_start_pattern_crn should be copied
        # Container collector of island elements
        parts_of_island = [FacePattern(init_face, self.face_start_pattern_crn_fw[0], self.face_start_pattern_crn_fw)]
        init_face.tag = False

        init_idx = init_face.index
        temp = []  # Container for get elements from loop from parts_of_island

        while parts_of_island:  # Blank list == all faces of the island taken
            for f in parts_of_island:
                for crn_idx, crn in enumerate(f):  # Running through all the neighboring faces
                    shared_crn = crn.link_loop_radial_prev
                    if shared_crn == crn:
                        continue

                    shared_face = shared_crn.face
                    if not shared_face.tag:
                        continue

                    # Three and more linked faces case when mesh island calc (maybe in uv also).
                    if shared_face.index != init_idx:
                        continue

                    f[crn_idx] = len(shared_face.loops)
                    temp.append(FacePattern.calc_fw(shared_face, shared_crn))
                    shared_face.tag = False


            self.walked_island_from_init_face.append(parts_of_island)
            parts_of_island = temp
            temp = []

    def transfer_co_to(self, other: list[list[FacePattern]], other_uv):
        uv = self.island.umesh.uv
        for t_list_f, s_list_f in zip(self.walked_island_from_init_face, other):
            for t_f, s_f in zip(t_list_f, s_list_f):
                s_f.start_crn[other_uv].uv = t_f.start_crn[uv].uv
                for t_crn, s_crn in zip(t_f.ordered_corners, s_f.ordered_corners):
                    s_crn[other_uv].uv = t_crn[uv].uv

    def __eq__(self, other: 'typing.Self'):
        return np.array_equal(self.np_ngons_with_faces, other.np_ngons_with_faces)


class UNIV_OT_Stack_VIEW3D(bpy.types.Operator):
    bl_idname = "mesh.univ_stack"
    bl_label = "Stack"
    bl_description = "Topology stack islands to selected."
    bl_options = {'REGISTER', 'UNDO'}

    # noinspection PyTypeHints
    between_selected: bpy.props.BoolProperty(name='Between Selected', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH'

    def draw(self, context):
        self.layout.prop(self, 'between_selected', toggle=1)

    # def invoke(self, context, event):
    #     return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: UMeshes | None = None
        self.targets: list[StackIsland] = []
        self.transfer: list[StackIsland] = []
        self.counter: int = 0
        self.calc_selected: Callable = AdvIslands.calc_selected
        self.calc_non_selected: Callable = AdvIslands.calc_non_selected

    def execute(self, context) -> set[str]:
        self.counter = 0
        self.umeshes = UMeshes(report=self.report)
        self.umeshes.update_tag = False
        if not self.umeshes.sync and context.area.ui_type != 'UV':
            self.umeshes.set_sync(True)

        if self.between_selected:
            self.stack_selected_between()
        else:
            self.stack_transfer_to_target()
        if self.counter:
            self.report({'INFO'}, f'Found {self.counter} islands for stacking')

        return {'FINISHED'}

    # Between Selected
    def stack_selected_between(self):
        self.targets: list[StackIsland] = []

        for sort_stack_islands in self.islands_preprocessing_selected_between():
            for target in sort_stack_islands:
                if target.island.tag:
                    target.island.tag = False
                    if not any(i.island.tag for i in sort_stack_islands):
                        break
                    target.calc_walked_reference_island_fw_and_for_bw()
                    for transfer in sort_stack_islands:
                        if transfer.island.tag:
                            if res := target.calc_transfer_stack_island_fw(transfer):
                                target.transfer_co_to(res, transfer.island.umesh.uv)

                                transfer.island.mark_seam()
                                transfer.island.umesh.update_tag = True
                                transfer.island.tag = False
                                self.counter += 1
                            transfer.island.set_tag(False)  # TODO: Check when else

        self.umeshes.update(info_type={'WARNING'}, info='No found islands for stacking')

    def islands_preprocessing_selected_between(self, stack_type: typing.Type[T] = StackIsland) -> list[list[T]]:
        for umesh in reversed(self.umeshes):
            selected = self.calc_selected(umesh)
            if not selected:
                self.umeshes.umeshes.remove(umesh)
                continue

            if isinstance(selected, utypes.MeshIslands):
                selected = selected.to_adv_islands()

            selected.indexing()

            for sel_isl in selected:
                stack_isl = stack_type(sel_isl)
                stack_isl.preprocessing()
                self.targets.append(stack_isl)

        if not self.targets:
            self.report({'WARNING'}, 'Not found selected islands')
            return []

        if not (sort_stack_islands_groups := utils.true_groupby(self.targets)):
            self.report({'WARNING'}, 'Islands have different set and number of polygons')
            return []
        return sort_stack_islands_groups

    def stack_transfer_to_target(self):
        self.targets: list[StackIsland] = []
        self.transfer: list[StackIsland] = []

        sorted_target_islands_with_transfer = self.islands_preprocessing_target_and_transfer()

        for target, stacks_transfer in sorted_target_islands_with_transfer:
            for transfer in stacks_transfer:
                if transfer.island.tag:
                    if res := target.calc_transfer_stack_island_fw(transfer):
                        target.transfer_co_to(res, transfer.island.umesh.uv)

                        transfer.island.mark_seam()
                        transfer.island.umesh.update_tag = True
                        transfer.island.tag = False
                        self.counter += 1
                    transfer.island.set_tag(False)

        self.umeshes.update(info_type={'WARNING'}, info='No found islands for stacking')

    def islands_preprocessing_target_and_transfer(self, stack_type=StackIsland):
        for umesh in reversed(self.umeshes):
            # TODO: With expanded selection, you can calculate islands via calc_visible
            #  and add them to selected and non_selected. This does not work with calc_selected.
            selected = self.calc_selected(umesh)
            non_selected = self.calc_non_selected(umesh)

            if not selected and not non_selected:
                self.umeshes.umeshes.remove(umesh)
                continue

            if isinstance(selected, utypes.MeshIslands):
                selected = selected.to_adv_islands()
                non_selected = non_selected.to_adv_islands()

            if not selected:
                proxi = non_selected
            elif not non_selected:
                proxi = selected
            else:
                proxi = AdvIslands(non_selected.islands + selected.islands, umesh)

            proxi.indexing()

            for sel_isl in selected:
                stack_isl = stack_type(sel_isl)
                stack_isl.preprocessing()
                stack_isl.calc_walked_reference_island_fw_and_for_bw()
                self.targets.append(stack_isl)

            for non_sel_isl in non_selected:
                stack_isl = stack_type(non_sel_isl)
                stack_isl.preprocessing()
                self.transfer.append(stack_isl)

        if not self.targets:
            self.report({'WARNING'}, 'Not found target islands')
            return []

        if not self.transfer:
            self.report({'WARNING'}, 'Not found transfer islands')
            return []

        if not (sort_stack_islands := self.sort_stack_islands_target_with_transfer()):
            self.report({'WARNING'}, 'Islands have different set and number of polygons')
        return sort_stack_islands

    def sort_stack_islands_target_with_transfer(self):
        sorted_groups: list[tuple[StackIsland, list[StackIsland]]] = []
        for tar in self.targets:
            group: list[StackIsland] = []
            for transfer_isl in self.transfer:
                if tar == transfer_isl:
                    group.append(transfer_isl)
            if group:
                sorted_groups.append((tar, group))
        return sorted_groups


class UNIV_OT_Stack(UNIV_OT_Stack_VIEW3D):
    bl_idname = "uv.univ_stack"
