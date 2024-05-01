import math
from mathutils import Vector, Matrix


class BBox:
    @classmethod
    def calc_bbox(cls, coords):
        xmin = math.inf
        xmax = -math.inf
        ymin = math.inf
        ymax = -math.inf

        for x, y in coords:
            if xmin > x:
                xmin = x
            if xmax < x:
                xmax = x
            if ymin > y:
                ymin = y
            if ymax < y:
                ymax = y
        return cls(xmin, xmax, ymin, ymax)

    @classmethod
    def calc_bbox_uv(cls, group, uv_layers):
        xmin = math.inf
        xmax = -math.inf
        ymin = math.inf
        ymax = -math.inf

        for face in group:
            for loop in face.loops:
                x, y = loop[uv_layers].uv
                if xmin > x:
                    xmin = x
                if xmax < x:
                    xmax = x
                if ymin > y:
                    ymin = y
                if ymax < y:
                    ymax = y

        return cls(xmin, xmax, ymin, ymax)

    @classmethod
    def calc_bbox_uv_loops(cls, group, uv_layers):
        xmin = math.inf
        xmax = -math.inf
        ymin = math.inf
        ymax = -math.inf

        for loop in group:
            x, y = loop[uv_layers].uv
            if xmin > x:
                xmin = x
            if xmax < x:
                xmax = x
            if ymin > y:
                ymin = y
            if ymax < y:
                ymax = y
        return cls(xmin, xmax, ymin, ymax)

    @classmethod
    def init_from_minmax(cls, minimum, maximum):
        bbox = cls(minimum[0], maximum[0], minimum[1], maximum[1])
        bbox.sanitize()
        return bbox

    def __init__(self, xmin=math.inf, xmax=-math.inf, ymin=math.inf, ymax=-math.inf):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def __str__(self):
        return f"xmin={self.xmin:.6}, xmax={self.xmax:.6}, ymin={self.ymin:.6}, ymax={self.ymax:.6}, width={self.width:.6}, height={self.height:.6}"

    @property
    def is_valid(self) -> bool:
        return (self.xmin <= self.xmax) and (self.ymin <= self.ymax)

    def is_valid_for_div(self) -> bool:
        return self.min_length > 0

    @property
    def max(self):
        return Vector((self.xmax, self.ymax))

    @property
    def min(self):
        return Vector((self.xmin, self.ymin))

    @property
    def left_upper(self):
        return Vector((self.xmin, self.ymax))

    @property
    def left_bottom(self):
        return Vector((self.xmin, self.ymin))

    @property
    def right_bottom(self):
        return Vector((self.xmax, self.ymin))

    @property
    def right_upper(self):
        return Vector((self.xmax, self.ymax))

    @property
    def upper(self):
        return Vector(((self.xmin + self.xmax) * 0.5, self.ymax))

    @property
    def bottom(self):
        return Vector(((self.xmin + self.xmax) * 0.5, self.ymin))

    @property
    def left(self):
        return Vector((self.xmin, (self.ymin + self.ymax) * 0.5))

    @property
    def right(self):
        return Vector((self.xmax, (self.ymin + self.ymax) * 0.5))

    @property
    def center(self):
        return Vector(((self.xmin + self.xmax) * 0.5, (self.ymin + self.ymax) * 0.5))

    @center.setter
    def center(self, new_center):
        delta = new_center - self.center
        self.move(delta)

    def move(self, delta):
        x, y = delta
        self.xmin += x
        self.xmax += x
        self.ymin += y
        self.ymax += y

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def max_length(self):
        return max(self.width, self.height)

    @property
    def min_length(self):
        return min(self.width, self.height)

    @property
    def half_width(self) -> float:
        return (self.xmax - self.xmin) * 0.5

    @property
    def half_height(self) -> float:
        return (self.ymax - self.ymin) * 0.5

    @property
    def area(self):
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        return (self.xmax <= self.xmin) or (self.ymax <= self.ymin)

    @property
    def cent_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def cent_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0

    @property
    def half_size_x(self) -> float:
        return (self.xmax - self.xmin) * 0.5

    @property
    def half_size_y(self) -> float:
        return (self.ymax - self.ymin) * 0.5

    @property
    def perimetr(self):
        return self.width * 2 + self.height * 2

    @property
    def diagonal(self):
        return math.sqrt(self.width ** 2 + self.height ** 2)

    def union(self, other):
        if self.xmin > other.xmin:
            self.xmin = other.xmin
        if self.xmax < other.xmax:
            self.xmax = other.xmax
        if self.ymin > other.ymin:
            self.ymin = other.ymin
        if self.ymax < other.ymax:
            self.ymax = other.ymax
        return self

    def sanitize(self):
        if self.xmin > self.xmax:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.ymin > self.ymax:
            self.ymin, self.ymax = self.ymax, self.ymin
        # assert self.is_valid
        return self

    def do_minmax_v(self, xy):
        if xy[0] < self.xmin:
            self.xmin = xy[0]
        if xy[0] > self.xmax:
            self.xmax = xy[0]
        if xy[1] < self.ymin:
            self.ymin = xy[1]
        if xy[1] > self.ymax:
            self.ymax = xy[1]

    def clamp(self, xmin=0, ymin=0, xmax=1, ymax=1):
        if self.xmin < xmin:
            self.xmin = xmin
        if self.ymin < ymin:
            self.ymin = ymin
        if self.xmax > xmax:
            self.xmax = xmax
        if self.ymax > ymax:
            self.ymax = ymax

    def translate(self, delta):
        self.xmin, self.ymin = self.min + delta
        self.xmax, self.ymax = self.max + delta
        return self

    def rotate_expand(self, angle):
        center = self.center
        rot_matrix = Matrix.Rotation(-angle, 2)

        corner = self.right_upper - center
        corner_rot = corner @ rot_matrix
        corner_max = Vector((abs(corner_rot[0]), abs(corner_rot[1])))

        corner.y *= -1
        corner_rot = corner @ rot_matrix
        corner_max[0] = max(corner_max[0], abs(corner_rot[0]))
        corner_max[1] = max(corner_max[1], abs(corner_rot[1]))

        self.xmin = center[0] - corner_max[0]
        self.xmax = center[0] + corner_max[0]
        self.ymin = center[1] - corner_max[1]
        self.ymax = center[1] + corner_max[1]

        return self

    def scale(self, scale):
        center = self.center
        self.xmin, self.ymin = (self.min - center) * scale + center
        self.xmax, self.ymax = (self.max - center) * scale + center
        return self.sanitize()

    def update(self, coords):
        for x, y in coords:
            if x < self.xmin:
                self.xmin = x
            if x > self.xmax:
                self.xmax = x
            if y < self.ymin:
                self.ymin = y
            if y > self.ymax:
                self.ymax = y

    def isect_x(self, x) -> bool:
        if x < self.xmin:
            return False
        if x > self.xmax:
            return False
        return True

    def isect_y(self, y) -> bool:
        if y < self.ymin:
            return False
        if y > self.ymax:
            return False
        return True

    def isect_pt_v(self, xy: Vector) -> bool:
        if xy[0] < self.xmin:
            return False
        if xy[0] > self.xmax:
            return False
        if xy[1] < self.ymin:
            return False
        if xy[1] > self.ymax:
            return False
        return True

    def length_x(self, x) -> float:
        if x < self.xmin:
            return self.xmin - x
        if x > self.xmax:
            return x - self.xmax
        return 0.0

    def length_y(self, y) -> float:
        if y < self.ymin:
            return self.ymin - y
        if y > self.ymax:
            return y - self.ymax
        return 0

    def isect_segment(self, s1: Vector, s2: Vector) -> bool:
        from mathutils.geometry import intersect_line_line_2d as ll_isect
        return any((ll_isect(s1, s2, self.left_bottom, self.left_upper),
                    ll_isect(s1, s2, self.left_upper, self.right_upper),
                    ll_isect(s1, s2, self.right_upper, self.right_bottom),
                    ll_isect(s1, s2, self.right_bottom, self.left_bottom)))

    def isect_circle(self, xy: Vector, radius: float) -> bool:
        if self.xmin <= xy.x <= self.xmax:
            dx = 0
        else:
            dx = (self.xmin - xy.x) if (xy.x < self.xmin) else (xy.x - self.xmax)

        if self.ymin <= xy.y <= self.ymax:
            dy = 0
        else:
            dy = (self.ymin - xy.y) if (xy.y < self.ymin) else (xy.y - self.ymax)

        return dx * dx + dy * dy <= radius * radius

    def transform_pt_v(self, *dst: 'BBox', xy_src: list[float]) -> list[float]:
        xy_dst = [0.0, 0.0]
        xy_dst[0] = ((xy_src[0] - self.xmin) / (self.xmax - self.xmin))
        xy_dst[0] = dst.xmin + ((dst.xmax - dst.xmin) * xy_dst[0])

        xy_dst[1] = ((xy_src[1] - self.ymin) / (self.ymax - self.ymin))
        xy_dst[1] = dst.ymin + ((dst.ymax - dst.ymin) * xy_dst[1])

        return xy_dst

    def transform_calc_m4_pivot_min(self, dst: 'BBox') -> Matrix:
        matrix = Matrix.Identity(4)
        matrix[0][0] = self.width / dst.width
        matrix[1][1] = self.height / dst.height
        matrix[3][0] = (self.xmin - dst.xmin) * matrix[0][0]
        matrix[3][1] = (self.ymin - dst.ymin) * matrix[1][1]
        return matrix

    def pad(self, pad: Vector):
        self.xmin -= pad.x
        self.ymin -= pad.y
        self.xmax += pad.x
        self.ymax += pad.y

    def pad_y(self, boundary_size, pad_min, pad_max):
        assert (pad_max >= 0.0)
        assert (pad_min >= 0.0)
        assert (boundary_size > 0.0)

        total_pad = pad_max + pad_min
        if total_pad == 0.0:
            return

        total_extend = self.width * total_pad / (boundary_size - total_pad)
        self.ymax += total_extend * (pad_max / total_pad)
        self.ymin -= total_extend * (pad_min / total_pad)

    def resize_x(self, x):
        self.xmin = self.cent_x - (x * 0.5)
        self.xmax = self.xmin + x

    def resize_y(self, y):
        self.ymin = self.cent_y - (y * 0.5)
        self.ymax = self.ymin + y

    def resize(self, xy: Vector):
        self.xmin = self.cent_x - (xy[0] * 0.5)
        self.ymin = self.cent_y - (xy[1] * 0.5)
        self.xmax = self.xmin + xy[0]
        self.ymax = self.ymin + xy[1]

    def interp(self, bbox_b: 'BBox', fac):
        ifac = 1.0 - fac
        bbox_r = BBox()
        bbox_r.xmin = self.xmin * ifac + bbox_b.xmin * fac
        bbox_r.xmax = self.xmax * ifac + bbox_b.xmax * fac
        bbox_r.ymin = self.ymin * ifac + bbox_b.ymin * fac
        bbox_r.ymax = self.ymax * ifac + bbox_b.ymax * fac
        return bbox_r

    def clamp_pt(self, xy: list[float, float]) -> bool:
        changed = False
        if xy[0] < self.xmin:
            xy[0] = self.xmin
            changed = True

        if xy[0] > self.xmax:
            xy[0] = self.xmax
            changed = True

        if xy[1] < self.ymin:
            xy[1] = self.ymin
            changed = True

        if xy[1] > self.ymax:
            xy[1] = self.ymax
            changed = True

        return changed

    def clamp_other(self, rect_bounds: 'BBox', r_xy: list[float, float]):
        changed = False

        r_xy[0] = 0.0
        r_xy[1] = 0.0

        if self.xmax > rect_bounds.xmax:
            ofs = rect_bounds.xmax - self.xmax
            self.xmin += ofs
            self.xmax += ofs
            r_xy[0] += ofs
            changed = True

        if self.xmin < rect_bounds.xmin:
            ofs = rect_bounds.xmin - self.xmin
            self.xmin += ofs
            self.xmax += ofs
            r_xy[0] += ofs
            changed = True

        if self.ymin < rect_bounds.ymin:
            ofs = rect_bounds.ymin - self.ymin
            self.ymin += ofs
            self.ymax += ofs
            r_xy[1] += ofs
            changed = True

        if self.ymax > rect_bounds.ymax:
            ofs = rect_bounds.ymax - self.ymax
            self.ymin += ofs
            self.ymax += ofs
            r_xy[1] += ofs
            changed = True

        return changed

    def compare(self, other: 'BBox', threshold):
        if abs(self.xmin - other.xmin) < threshold:
            if abs(self.xmax - other.xmax) < threshold:
                if abs(self.ymin - other.ymin) < threshold:
                    if abs(self.ymax - other.ymax) < threshold:
                        return True
        return False

    def isect(self, other: 'BBox') -> 'BBox | None':
        xmin = self.xmin if (self.xmin > other.xmin) else other.xmin
        xmax = self.xmax if (self.xmax < other.xmax) else other.xmax
        ymin = self.ymin if (self.ymin > other.ymin) else other.ymin
        ymax = self.ymax if (self.ymax < other.ymax) else other.ymax

        if xmax >= xmin and ymax >= ymin:
            return BBox(xmin, xmax, ymin, ymax)

    def isect_rect_y(self, other: 'BBox') -> 'tuple[float, float] | None':
        ymin = self.ymin if (self.ymin > other.ymin) else other.ymin
        ymax = self.ymax if (self.ymax < other.ymax) else other.ymax

        if ymax >= ymin:
            return ymin, ymax

    def isect_rect_x(self, other: 'BBox') -> 'tuple[float, float] | None':
        xmin = self.xmin if (self.xmin > other.xmin) else other.xmin
        xmax = self.xmax if (self.xmax < other.xmax) else other.xmax

        if xmax >= xmin:
            return xmin, xmax

    def __contains__(self, pt_or_bbox) -> bool:
        if isinstance(BBox, pt_or_bbox):
            bbox = pt_or_bbox
            return (self.xmin <= bbox.xmin) and (self.xmax >= bbox.xmax) and \
                   (self.ymin <= bbox.ymin) and (self.ymax >= bbox.ymax)  # noqa

        x, y = pt_or_bbox
        return self.xmin <= x <= self.xmax and \
               self.ymin <= y <= self.ymax  # noqa

    def __eq__(self, other: 'BBox'):
        return self.min == other.min and self.max == other.max

    def __and__(self, other: 'BBox'):
        self.isect(other)

    def __or__(self, other: 'BBox'):
        self.union(other)
