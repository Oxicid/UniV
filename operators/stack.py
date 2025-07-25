# SPDX-FileCopyrightText: 2024 Oxicid
# SPDX-License-Identifier: GPL-3.0-or-later

if 'bpy' in locals():
    from .. import reload
    reload.reload(globals())

import bpy
import typing
import numpy as np

from .. import utils
from ..utils import linked_crn_to_vert_by_face_index
from .. import types
from ..types import AdvIsland, AdvIslands
from collections import deque, defaultdict
from collections.abc import Callable

from bmesh.types import BMFace, BMLoop

# TODO: Rename target to reference
T = typing.TypeVar('T')
def py_container_with_np_arrays_compare(a: deque[np.array], b: deque[np.array]):
    return len(a) == len(b) and all(np.array_equal(a[i], b[i]) for i in range(len(a)))

class FacePattern:
    def __init__(self, f, start_crn, ordered_corners):
        self.face: BMFace = f
        self.start_crn: BMLoop = start_crn
        self.ordered_corners: deque[BMLoop] = ordered_corners
        self.shared_crn_face_sizes: np.array = np.zeros(shape=len(ordered_corners), dtype='uint16')

    @classmethod
    def calc_init(cls, f, start_crn):
        return cls(f, start_crn, deque(f.loops))

    @classmethod
    def calc(cls, f, start_crn):
        linked = deque(f.loops)
        linked.rotate(-linked.index(start_crn))
        linked.popleft()
        return cls(f, start_crn, linked)

    @classmethod
    def calc_backward(cls, f, start_crn):
        linked = deque(reversed(f.loops))
        linked.rotate(-linked.index(start_crn))
        linked.popleft()
        return cls(f, start_crn, linked)

    def __str__(self):
        return f"FacePattern: shared crn face size = {self.shared_crn_face_sizes}"

    def __iter__(self) -> typing.Iterator[BMLoop]:
        return iter(self.ordered_corners)

    def __setitem__(self, key: int, value: int):
        self.shared_crn_face_sizes[key] = value

class StackIsland:  # TODO: Split for source and target islands
    def __init__(self, island):
        self.island: AdvIsland = island
        self.walked_island_from_init_face: list[list[FacePattern]] = []

        self.unique_faces: list[BMFace] = []

        self.face_start_pattern_crn: deque[BMLoop] = deque()
        self.face_start_pattern: deque[np.array] = deque()
        self.start_faces_patterns: list[list[deque[np.array], deque[BMLoop]]] = []

        self.face_start_pattern_crn_backward: deque[BMLoop] = deque()
        self.face_start_pattern_backward: deque[np.array] = deque()
        self.start_faces_patterns_backward: list[list[deque[np.array], deque[BMLoop]]] = []

        self.ngons_with_faces: defaultdict[int, list[BMFace]] = defaultdict(list)
        self.np_ngons_with_faces: np.array = np.array([], dtype='int32')

    def preprocessing(self):
        self.ensure_ngons_with_faces()
        self.calc_unique_faces()
        self.unique_faces.sort(key=lambda face: face.calc_area(), reverse=True)
        self.ngons_to_np()
        self.face_start_pattern, self.face_start_pattern_crn = self.calc_linked_corners_pattern(0)

    def ensure_ngons_with_faces(self):
        for f in self.island:
            self.ngons_with_faces[len(f.loops)].append(f)

    def calc_unique_faces(self) -> list[BMFace]:
        assert self.ngons_with_faces

        min_sequence: tuple[int | list[BMFace]] | tuple = ()
        for ngons_size, faces in self.ngons_with_faces.items():
            if not min_sequence:
                min_sequence = ngons_size, faces

            elif len(faces) < len(min_sequence[1]):
                min_sequence = ngons_size, faces
            elif len(faces) == len(min_sequence[1]) and ngons_size > min_sequence[0]:
                min_sequence = ngons_size, faces

        self.unique_faces = min_sequence[1]
        return self.unique_faces

    def ngons_to_np(self):
        np_ngons_with_faces = np.empty(shape=(len(self.ngons_with_faces), 2), dtype='int32')
        for idx, (f_size, faces) in enumerate(self.ngons_with_faces.items()):
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
            from ..types import bbox
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

    def calc_linked_corners_pattern(self, idx) -> list[deque[np.ndarray] | deque[BMLoop]]:
        init_face: BMFace = self.unique_faces[idx]
        face_idx = init_face.index

        face_start_pattern: deque[np.ndarray] = deque()  # [shared face size, linked face size]
        face_start_pattern_crn: deque[BMLoop] = deque(init_face.loops)

        unique_dist_idx = self.get_unique_start_crn_idx(face_start_pattern_crn)
        face_start_pattern_crn.rotate(-unique_dist_idx)

        for idx__, crn in enumerate(face_start_pattern_crn):
            linked_crn_face_size = []
            for crn_ in linked_crn_to_vert_by_face_index(crn):
                linked_crn_face_size.append(len(crn_.face.loops))

            shared_face_size = 0
            shared_crn = crn.link_loop_radial_prev
            if shared_crn != crn and shared_crn.face.index == face_idx:
                shared_face_size = len(shared_crn.face.loops)

            linked_crn_face_size.sort()
            face_start_pattern.append(np.array((shared_face_size, *linked_crn_face_size), dtype='uint16'))

        return [face_start_pattern, face_start_pattern_crn]

    def calc_all_linked_corners_pattern(self):
        start_faces_patterns: list[list[deque[np.ndarray] | deque[BMLoop]]] = []
        for idx, sequ in enumerate(self.unique_faces):
            # continue
            if not idx:
                continue
            start_faces_patterns.append(self.calc_linked_corners_pattern(idx))
        self.start_faces_patterns = start_faces_patterns

    def compared_matching_first_pattern(self, other: 'typing.Self'):
        if len(self.face_start_pattern) != len(other.face_start_pattern):
            return

        for _ in range(len(other.face_start_pattern) - 1):

            if py_container_with_np_arrays_compare(self.face_start_pattern, other.face_start_pattern):
                yield other.face_start_pattern_crn

            other.face_start_pattern.rotate(1)
            other.face_start_pattern_crn.rotate(1)

        if not other.start_faces_patterns:
            other.calc_all_linked_corners_pattern()

        for face_start_pattern, face_start_pattern_crn in other.start_faces_patterns:
            for _ in range(len(face_start_pattern) - 1):
                if py_container_with_np_arrays_compare(self.face_start_pattern, face_start_pattern):
                    yield face_start_pattern_crn

                face_start_pattern.rotate(1)
                face_start_pattern_crn.rotate(1)

    def calc_source_stack_island(self, source: 'typing.Self'):
        for source_pattern in self.compared_matching_first_pattern(source):
            source_island_walked: list[list[FacePattern]] = []
            source.island.set_tag(True)  # TODO: Tagging source_island_walked

            start_crn_ = source_pattern[0]
            init_face_pattern = FacePattern(start_crn_.face, start_crn_, source_pattern)
            parts_of_island: list[FacePattern] = [init_face_pattern]  # Container collector of island elements
            init_face_pattern.face.tag = False

            face_idx = init_face_pattern.face.index

            temp = []  # Container for get elements from loop from parts_of_island
            break_ = False

            for generation_of_shared_face in self.walked_island_from_init_face:
                for target_face__, source_face__ in zip(generation_of_shared_face, parts_of_island):
                    if len(target_face__.ordered_corners) != len(source_face__.ordered_corners):
                        break_ = True
                        break

                    for target_face_size, source_crn in zip(target_face__.shared_crn_face_sizes, source_face__.ordered_corners):
                        if target_face_size == 0:
                            continue

                        if (shared_crn := source_crn.link_loop_radial_prev) == source_crn:
                            break_ = True
                            break

                        shared_crn_face = shared_crn.face
                        if target_face_size != len(shared_crn_face.loops) or (not shared_crn_face.tag) or (shared_crn_face.index != face_idx):
                            break_ = True
                            break

                        shared_crn_face.tag = False
                        temp.append(FacePattern.calc(shared_crn_face, shared_crn))
                    if break_:
                        break

                source_island_walked.append(parts_of_island)
                parts_of_island = temp
                temp = []
                if break_:
                    break
            if not break_:
                return source_island_walked

    @staticmethod
    def pattern_to_str(pattern):
        text = ''
        for idx, p in enumerate(pattern):
            text += f'{idx}: {list(p)}, '
        return text

    def calc_walked_reference_island(self):
        """Create an island that saves a sequence of found faces and its shared corner"""
        self.island.set_tag(True)

        init_face = self.unique_faces[0]
        # For the future be careful, maybe face_start_pattern_crn should be copied
        parts_of_island = [FacePattern(init_face, self.face_start_pattern_crn[0], self.face_start_pattern_crn)]  # Container collector of island elements
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

                    f[crn_idx] = len(shared_face.loops)
                    temp.append(FacePattern.calc(shared_face, shared_crn))
                    shared_face.tag = False
                    assert shared_face.index == init_idx

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
    bl_description = "Stack to selected"
    bl_options = {'REGISTER', 'UNDO'}

    between_selected: bpy.props.BoolProperty(name='Between Selected', default=False)

    @classmethod
    def poll(cls, context):
        return context.mode == 'EDIT_MESH' and (obj := context.active_object) and obj.type == 'MESH'  # noqa # pylint:disable=used-before-assignment

    def draw(self, context):
        self.layout.prop(self, 'between_selected', toggle=1)

    # def invoke(self, context, event):
    #     return self.execute(context)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.umeshes: types.UMeshes | None = None
        self.targets: list[StackIsland] = []
        self.source: list[StackIsland] = []
        self.counter: int = 0
        self.calc_selected: Callable = AdvIslands.calc_selected
        self.calc_non_selected: Callable = AdvIslands.calc_non_selected

    def execute(self, context) -> set[str]:
        self.counter = 0
        self.umeshes = types.UMeshes(report=self.report)
        self.umeshes.update_tag = False
        if not self.umeshes.sync and context.area.ui_type != 'UV':
            self.umeshes.set_sync(True)

        if self.between_selected:
            self.stack_selected_between()
        else:
            self.stack_target_source()
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
                    target.calc_walked_reference_island()
                    for source in sort_stack_islands:
                        if source.island.tag:
                            if res := target.calc_source_stack_island(source):
                                target.transfer_co_to(res, source.island.umesh.uv)
                                source.island.umesh.update_tag = True
                                source.island.tag = False
                                self.counter += 1
                            source.island.set_tag(False)  # TODO: Check when else

        self.umeshes.update(info_type={'WARNING'}, info='No found islands for stacking')

    def islands_preprocessing_selected_between(self, stack_type: typing.Type[T] = StackIsland) -> list[list[T]]:
        for umesh in reversed(self.umeshes):
            selected = self.calc_selected(umesh)
            if not selected:
                self.umeshes.umeshes.remove(umesh)
                continue

            if isinstance(selected, types.MeshIslands):
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

    # Target Source
    def stack_target_source(self):
        self.targets: list[StackIsland] = []
        self.source: list[StackIsland] = []

        sort_stack_islands = self.islands_preprocessing_reference_and_source()

        for stack_target, stacks_source in sort_stack_islands:
            for stacks_source_isl in stacks_source:
                if stacks_source_isl.island.tag:
                    if res := stack_target.calc_source_stack_island(stacks_source_isl):
                        stack_target.transfer_co_to(res, stacks_source_isl.island.umesh.uv)
                        stacks_source_isl.island.umesh.update_tag = True
                        stacks_source_isl.island.tag = False
                        self.counter += 1
                    stacks_source_isl.island.set_tag(False)

        self.umeshes.update(info_type={'WARNING'}, info='No found islands for stacking')

    def islands_preprocessing_reference_and_source(self, stack_type=StackIsland):
        for umesh in reversed(self.umeshes):
            # TODO: With expanded selection, you can calculate islands via calc_visible
            #  and add them to selected and non_selected. This does not work with calc_selected.
            selected = self.calc_selected(umesh)
            non_selected = self.calc_non_selected(umesh)

            if not selected and not non_selected:
                self.umeshes.umeshes.remove(umesh)
                continue

            if isinstance(selected, types.MeshIslands):
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
                stack_isl.calc_walked_reference_island()
                self.targets.append(stack_isl)

            for non_sel_isl in non_selected:
                stack_isl = stack_type(non_sel_isl)
                stack_isl.preprocessing()
                self.source.append(stack_isl)

        if not self.targets:
            self.report({'WARNING'}, 'Not found target islands')
            return []

        if not self.source:
            self.report({'WARNING'}, 'Not found source islands')
            return []

        if not (sort_stack_islands := self.sort_stack_islands_target_source()):
            self.report({'WARNING'}, 'Islands have different set and number of polygons')
        return sort_stack_islands

    def sort_stack_islands_target_source(self):
        sorted_groups: list[tuple[StackIsland, list[StackIsland]]] = []
        for tar in self.targets:
            group: list[StackIsland] = []
            for source_isl in self.source:
                if tar == source_isl:
                    group.append(source_isl)
            if group:
                sorted_groups.append((tar, group))
        return sorted_groups

class UNIV_OT_Stack(UNIV_OT_Stack_VIEW3D):
    bl_idname = "uv.univ_stack"
