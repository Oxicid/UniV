import typing
from collections import defaultdict
from bmesh.types import BMLoop, BMLayerItem
from ..utils import prev_disc
from .. import utils

class LoopGroup:
    def __init__(self, uv: BMLayerItem):
        self.uv: BMLayerItem = uv
        self.corners: list[BMLoop] = []
        self.tag = True

    def is_cyclic_vert(self):
        if len(self.corners) > 1:
            return self.corners[-1].link_loop_next.vert == self.corners[0].vert

    def is_cyclic_crn(self):
        if len(self.corners) > 1:
            return self.corners[-1].link_loop_next == self.corners[0]

    def calc_loop_group(self, crn):
        crn.tag = False
        group = [crn]
        while True:
            if next_crn := self.next_walk_boundary(group[-1]):
                group.append(next_crn)
            else:
                break
        while True:
            if prev_crn := self.prev_walk_boundary(group[0]):
                group.insert(0, prev_crn)
            else:
                break
        self.corners = group
        return self

    def next_walk_boundary(self, crn):
        crn_next = crn.link_loop_next
        if crn_next.tag:
            crn_next.tag = False
            return crn_next

        bm_iter = crn_next
        while True:
            if (bm_iter := prev_disc(bm_iter)) == crn_next:
                break
            if bm_iter.tag and crn_next[self.uv].uv == bm_iter[self.uv].uv:
                bm_iter.tag = False
                return bm_iter

    def prev_walk_boundary(self, crn):
        crn_prev = crn.link_loop_prev
        if crn_prev.tag:
            crn_prev.tag = False
            return crn_prev

        bm_iter = crn
        while True:
            if (bm_iter := prev_disc(bm_iter)) == crn:
                break
            if bm_iter.link_loop_prev.tag and crn[self.uv].uv == bm_iter[self.uv].uv:
                bm_iter.link_loop_prev.tag = False
                return bm_iter.link_loop_prev

    def is_boundary_sync(self, crn):
        shared_crn = crn.link_loop_radial_prev
        if shared_crn == crn:
            return True
        if shared_crn.face.hide:  # Change
            return True
        return not (crn[self.uv].uv == shared_crn.link_loop_next[self.uv].uv and crn.link_loop_next[self.uv].uv == shared_crn[self.uv].uv)

    def is_boundary(self, crn):
        shared_crn = crn.link_loop_radial_prev  # We get a clockwise corner, but linked to the end of the current corner
        if shared_crn == crn:
            return True
        if not shared_crn.face.select:  # Change
            return True
        return not (crn[self.uv].uv == shared_crn.link_loop_next[self.uv].uv and crn.link_loop_next[self.uv].uv == shared_crn[self.uv].uv)

    def calc_shared_group(self) -> 'typing.Self':
        shared_group = []
        for crn in reversed(self.corners):
            shared_group.append(crn.link_loop_radial_prev)
        lg = LoopGroup(self.uv)
        lg.corners = shared_group
        return lg

    def boundary_tag_by_face_index(self, crn: BMLoop):
        shared_crn = crn.link_loop_radial_prev
        if shared_crn == crn:
            crn.tag = False
            return

        # if shared_crn.face.index in (-1, crn.face.index):
        if shared_crn.face.index == -1:
            crn.tag = False
            return

        crn.tag = not (crn[self.uv].uv == shared_crn.link_loop_next[self.uv].uv and crn.link_loop_next[self.uv].uv == shared_crn[self.uv].uv)

    def boundary_tag(self, crn: BMLoop):
        shared_crn = crn.link_loop_radial_prev
        if shared_crn == crn:
            crn.tag = False
            return
        if not crn[self.uv].select_edge:
            crn.tag = False
            return
        if not shared_crn.face.select:  # Change
            crn.tag = False
            return
        crn.tag = not (crn[self.uv].uv == shared_crn.link_loop_next[self.uv].uv and crn.link_loop_next[self.uv].uv == shared_crn[self.uv].uv)

    def boundary_tag_sync(self, crn: BMLoop):
        shared_crn = crn.link_loop_radial_prev
        if shared_crn == crn:
            crn.tag = False
            return
        if not crn.edge.select:
            crn.tag = False
            return
        if shared_crn.face.hide:  # Change
            crn.tag = False
            return
        crn.tag = not (crn[self.uv].uv == shared_crn.link_loop_next[self.uv].uv and crn.link_loop_next[self.uv].uv == shared_crn[self.uv].uv)

    @staticmethod
    def calc_island_index_for_stitch(island) -> defaultdict[int | list[BMLoop]]:
        islands_for_stitch = defaultdict(list)
        for f in island:
            for crn in f.loops:
                if crn.tag:
                    crn.tag = False
                    shared_crn = crn.link_loop_radial_prev
                    islands_for_stitch[shared_crn.face.index].append(crn)
        return islands_for_stitch

    def tagging(self, island):
        func: typing.Callable = self.boundary_tag_sync if utils.sync() else self.boundary_tag
        for f in island:
            for crn in f.loops:
                func(crn)

    def calc_first(self, island, selected=True):
        if selected:
            self.tagging(island)
        else:
            for f__ in island:
                for crn__ in f__.loops:
                    self.boundary_tag_by_face_index(crn__)

        indexes = self.calc_island_index_for_stitch(island)
        for k, corner_edges in indexes.items():
            for _crn in corner_edges:
                _crn.tag = True

            crn_edges = (__crn for __crn in corner_edges if __crn.tag)

            for crn_edge in crn_edges:
                loop_group = self.calc_loop_group(crn_edge)

                yield loop_group

                if loop_group.tag:
                    if len(loop_group) != len(corner_edges):
                        for _crn in corner_edges:
                            _crn.tag = False
                    break

    def __iter__(self):
        return iter(self.corners)

    def __getitem__(self, idx) -> BMLoop:
        return self.corners[idx]

    def __len__(self):
        return len(self.corners)

    def __bool__(self):
        return bool(self.corners)

    def __str__(self):
        return f'Corner Edge count = {len(self.corners)}'
