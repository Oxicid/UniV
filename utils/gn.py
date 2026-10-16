import bpy
import typing
import contextlib


class GlobalTree:
    tree: "None | bpy.types.GeometryNodeTree" = None

    @contextlib.contextmanager
    def __call__(self, name: str) -> "typing.Generator[bpy.types.GeometryNodeTree] | typing.Any":
        prev = GlobalTree.tree
        GlobalTree.tree = bpy.data.node_groups.new(type="GeometryNodeTree", name=name)

        yield GlobalTree.tree

        GlobalTree.tree = prev


gt = GlobalTree()


def fproperty(funcs):
    return property(*funcs())


def _loc_format(loc):
    return [(0 if l is None else l) for l in loc]


class Node(GlobalTree):
    bnode: "bpy.types.GeometryNodeTree"

    @fproperty
    def label():  # noqa
        def getter(self):
            return self.bnode.label

        def setter(self, name):
            self.bnode.label = name

        return getter, setter

    @fproperty
    def location():  # noqa
        def getter(self):
            return self.bnode.location

        def setter(self, loc):
            self.bnode.location = loc

        return getter, setter

    @fproperty
    def hide():  # noqa
        def getter(self):
            return self.bnode.hide

        def setter(self, state):
            self.bnode.hide = state

        return getter, setter

    @fproperty
    def width():  # noqa
        def getter(self):
            return self.bnode.width

        def setter(self, w):
            self.bnode.width = w

        return getter, setter

    @fproperty
    def height():  # noqa
        def getter(self):
            return self.bnode.height

        def setter(self, h):
            self.bnode.height = h

        return getter, setter

    def create_SeparateXYZ(self, loc=(None, None)):
        separate_xyz = SeparateXYZ(loc=loc)
        self >> separate_xyz  # noqa
        return separate_xyz

    @property
    def Input(self):
        return Socket(self, 0, is_output=False)

    @property
    def Output(self):
        return Socket(self, 0, is_output=True)

    def __rshift__(self, other):
        _ = self.Output >> other


class SimpleNode(Node):
    _node_name: str = ""

    def __init__(self, loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeMenuSwitch" = self.__create(loc)
        self.orig_loc = loc

    def __create(self, loc):
        new_node: "bpy.types.GeometryNodeMenuSwitch" = self.tree.nodes.new(self._node_name)
        new_node.location = _loc_format(loc)
        return new_node


class Output(Node):
    in_out = "OUTPUT"

    def __init__(self, loc=(None, None), add_default_socket=True):
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(add_default_socket, loc)
        self.orig_loc = loc

    def __create(self, add_default_socket, loc):
        in_out_node = self.tree.nodes.new(f"NodeGroup{self.in_out.capitalize()}")
        in_out_node.location = _loc_format(loc)
        if add_default_socket:
            self.add_default_socket()
        return in_out_node

    def add_float(self, name, default=None, min=None, max=None, subtype=None, force_non_field=None, attribute_domain=None):  # noqa # fmt: skip
        factor_socket = self.tree.interface.new_socket(
            name=name, in_out=self.in_out, socket_type="NodeSocketFloat"
        )
        if default is not None:
            factor_socket.default_value = default
        if min is not None:
            factor_socket.min_value = min
        if max is not None:
            factor_socket.max_value = max
        if subtype is not None:
            factor_socket.subtype = subtype
        if attribute_domain is not None:
            factor_socket.attribute_domain = attribute_domain
        if force_non_field is not None:
            factor_socket.force_non_field = force_non_field

    def add_vector(self, name, default=None, min=None, max=None, subtype=None, force_non_field=None, attribute_domain=None):  # noqa # fmt: skip
        factor_socket = self.tree.interface.new_socket(
            name=name, in_out=self.in_out, socket_type="NodeSocketVector"
        )
        if default is not None:
            factor_socket.default_value = default
        if min is not None:
            factor_socket.min_value = min
        if max is not None:
            factor_socket.max_value = max
        if subtype is not None:
            factor_socket.subtype = subtype
        if attribute_domain is not None:
            factor_socket.attribute_domain = attribute_domain
        if force_non_field is not None:
            factor_socket.force_non_field = force_non_field

    def add_str(self, name: str | None, default: str | None =None, subtype=None, force_non_field=None, attribute_domain=None):  # noqa # fmt: skip
        factor_socket = self.tree.interface.new_socket(
            name=name, in_out=self.in_out, socket_type="NodeSocketString"
        )
        if default is not None:
            factor_socket.default_value = default
        if subtype is not None:
            factor_socket.subtype = subtype
        if attribute_domain is not None:
            factor_socket.attribute_domain = attribute_domain
        if force_non_field is not None:
            factor_socket.force_non_field = force_non_field

    def add_bool(self, name, default=None, subtype=None, force_non_field=None, attribute_domain=None):  # noqa # fmt: skip
        factor_socket = self.tree.interface.new_socket(
            name=name, in_out=self.in_out, socket_type="NodeSocketBool"
        )
        if default is not None:
            factor_socket.default_value = default
        if subtype is not None:
            factor_socket.subtype = subtype
        if attribute_domain is not None:
            factor_socket.attribute_domain = attribute_domain
        if force_non_field is not None:
            factor_socket.force_non_field = force_non_field

    def add_menu(self, name: str | None, default: str | None =None, subtype=None, force_non_field=None, attribute_domain=None):  # noqa # fmt: skip
        factor_socket = self.tree.interface.new_socket(
            name=name, in_out=self.in_out, socket_type="NodeSocketMenu"
        )
        if default is not None:
            factor_socket.default_value = default
        if subtype is not None:
            factor_socket.subtype = subtype
        if attribute_domain is not None:
            factor_socket.attribute_domain = attribute_domain
        if force_non_field is not None:
            factor_socket.force_non_field = force_non_field

    def add_geometry(self, name="Geometry"):
        self.tree.interface.new_socket(
            name=name, in_out=self.in_out, socket_type="NodeSocketGeometry"
        )

    def add_default_socket(self):
        if not any(
            sk.socket_type == "NodeSocketGeometry"
            for sk in self.tree.interface.items_tree
            if sk.in_out == self.in_out
        ):
            self.add_geometry()

    def __getitem__(self, item: int | str):
        return Socket(self, item, is_output=self.in_out == "INPUT")


class Input(Output):
    in_out = "INPUT"


class Group(Node):
    # noinspection PyMissingConstructor
    def __init__(self, loc=(None, None)):
        self.bnode = self.__create(loc)

    def __create(self, loc):
        menu_switch: "bpy.types.GeometryNodeMenuSwitch" = self.tree.nodes.new("GeometryNodeGroup")

        menu_switch.location = _loc_format(loc)
        return menu_switch

    @fproperty
    def node_tree():  # noqa
        def getter(self):
            return self.bnode.node_tree

        def setter(self, tree_):
            self.bnode.node_tree = tree_

        return getter, setter


class Socket(GlobalTree):
    def __init__(self, node, socket_idx, is_output):
        self.node: Node = node
        self.socket_idx = socket_idx
        self.is_output = is_output
        self.outputs = node.bnode.outputs if is_output else None
        self.inputs = None if is_output else node.bnode.inputs

    def __call__(self, *node_with_sk_idx):
        if len(node_with_sk_idx) == 1:
            self.node = node_with_sk_idx[0]
            self.socket_idx = 0

        else:
            assert isinstance(node_with_sk_idx[1], int)
            self.node, self.socket_idx = node_with_sk_idx
        return self

    def __rshift__(self, other: "Socket | Node | tuple"):
        assert isinstance(other, (Socket, Node, tuple)), f"Given {type(other)!r}"
        assert self.is_output

        if isinstance(other, Socket):
            sk_output = self.outputs[self.socket_idx]
            sk_input = other.inputs[other.socket_idx]
            self.tree.links.new(sk_output, sk_input)
        else:
            if isinstance(other, tuple):
                node, socket_idx = other
                assert isinstance(node, Node)
                assert isinstance(socket_idx, int)

                sk_output = self.outputs[self.socket_idx]
                sk_input = node.bnode.inputs[socket_idx]  # noqa
                self.tree.links.new(sk_output, sk_input)
            else:
                sk_output = self.outputs[self.socket_idx]  # noqa
                sk_input = other.bnode.inputs[0]  # noqa
                self.tree.links.new(sk_output, sk_input)  # noqa


class Reroute(SimpleNode):
    _node_name = "NodeReroute"


class Index(SimpleNode):
    _node_name = "GeometryNodeInputIndex"


class CornersOfEdge(Node):
    def __init__(self, idx, loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeMenuSwitch" = self.__create(idx, loc)
        self.orig_loc = loc

    def __create(self, idx, loc):
        crn_of_edge = self.tree.nodes.new("GeometryNodeCornersOfEdge")
        crn_of_edge.inputs[2].default_value = idx  # Sort Index
        crn_of_edge.location = _loc_format(loc)
        return crn_of_edge

    @fproperty
    def Index():  # noqa
        def getter(self):
            return Socket(self, 2, is_output=False)

        def setter(self, idx):
            self.node.inputs[2].default_value = idx

        return getter, setter

    def next_in_face(self, offset=1, loc=(None, None)):
        crn_in_face = CornerInFace(offset=offset, loc=loc)
        self >> crn_in_face
        return crn_in_face


class CornerInFace(Node):
    def __init__(self, offset=1, loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeMenuSwitch" = self.__create(offset, loc)
        self.orig_loc = loc

    def __create(self, offset, loc):
        crn_of_edge = self.tree.nodes.new("GeometryNodeOffsetCornerInFace")
        crn_of_edge.inputs[1].default_value = offset  # Offset
        crn_of_edge.location = _loc_format(loc)
        return crn_of_edge

    @fproperty
    def Offset():  # noqa
        def getter(self):
            return Socket(self, 1, is_output=False)

        def setter(self, idx):
            self.node.inputs[1].default_value = idx

        return getter, setter


class EvaluateAtIndex(Node):
    def __init__(self, data_type="FLOAT_VECTOR", domain="CORNER", loc=(None, None)):  # noqa

        self.bnode: "bpy.types.GeometryNodeMenuSwitch" = self.__create(data_type, domain, loc)
        self.orig_loc = loc

    def __create(self, data_type, domain, loc):
        eval_at_idx = self.tree.nodes.new("GeometryNodeFieldAtIndex")
        eval_at_idx.data_type = data_type
        eval_at_idx.domain = domain
        eval_at_idx.location = _loc_format(loc)
        return eval_at_idx

    @property
    def Index(self):
        return Socket(self, 1, is_output=False)


class Compare(Node):
    def __init__(
        self, operation="NOT_EQUAL", data_type="VECTOR", mode="ELEMENT", loc=(None, None)
    ):  # noqa
        self.bnode: "bpy.types.GeometryNodeMenuSwitch" = self.__create(
            operation, data_type, mode, loc
        )
        self.orig_loc = loc

    def __create(self, operation, data_type, mode, loc):
        eval_at_idx = self.tree.nodes.new("FunctionNodeCompare")
        eval_at_idx.operation = operation
        eval_at_idx.data_type = data_type
        eval_at_idx.mode = mode
        eval_at_idx.location = _loc_format(loc)
        return eval_at_idx

    @fproperty
    def A():  # noqa
        def getter(self):
            assert self.bnode.data_type == "VECTOR", "Not implement for other types"
            vec_input_index = 0 if bpy.app.version >= (5, 2, 0) else 4
            return Socket(self, vec_input_index, is_output=False)

        def setter(self, val):
            vec_input_index = 0 if bpy.app.version >= (5, 2, 0) else 4
            assert self.bnode.data_type == "VECTOR", "Not implement for other types"
            self.bnode.inputs[vec_input_index].default_value = val

        return getter, setter

    @fproperty
    def B():  # noqa
        def getter(self):
            assert self.bnode.data_type == "VECTOR", "Not implement for other types"
            vec_input_index = (0 if bpy.app.version >= (5, 2, 0) else 4)+1
            return Socket(self, vec_input_index, is_output=False)

        def setter(self, val):
            vec_input_index = (0 if bpy.app.version >= (5, 2, 0) else 4)+1
            assert self.bnode.data_type == "VECTOR", "Not implement for other types"
            self.bnode.inputs[vec_input_index].default_value = val

        return getter, setter

    @fproperty
    def Threshold():  # noqa
        def getter(self):
            assert self.bnode.data_type == "VECTOR", "Not implement for other types"
            return Socket(self, 2 if bpy.app.version >= (5, 2, 0) else 12, is_output=False)

        def setter(self, val):
            assert self.bnode.data_type == "VECTOR", "Not implement for other types"
            self.bnode.inputs[2 if bpy.app.version >= (5, 2, 0) else 12].default_value = val

        return getter, setter


class BooleanMath(Node):
    def __init__(self, operation="NOT_EQUAL", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeMenuSwitch" = self.__create(operation, loc)
        self.orig_loc = loc

    def __create(self, operation, loc):
        eval_at_idx = self.tree.nodes.new("FunctionNodeBooleanMath")
        eval_at_idx.operation = operation
        eval_at_idx.location = _loc_format(loc)
        return eval_at_idx

    @fproperty
    def Threshold():  # noqa
        def getter(self):
            assert self.data_type == "VECTOR", "Not implement for other types"
            return Socket(self, 2 if bpy.app.version >= (5, 2, 0) else 12, is_output=False)

        def setter(self, val):
            assert self.data_type == "VECTOR", "Not implement for other types"
            self.node.inputs[2 if bpy.app.version >= (5, 2, 0) else 12].default_value = val

        return getter, setter


class MenuSwitch(Node):
    def __init__(self, enums, type: str = "INT", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeMenuSwitch" = self.__create(enums, type, loc)
        self.type = type
        self.enums = enums
        self.orig_loc = loc

    def __create(self, enums, typ, loc):
        menu_switch: "bpy.types.GeometryNodeMenuSwitch" = self.tree.nodes.new(
            "GeometryNodeMenuSwitch"
        )
        menu_switch.data_type = typ
        menu_switch.enum_items.clear()

        for idx, item in enumerate(enums):
            menu_switch.enum_items.new(item)
            menu_switch.inputs[idx + 1].default_value = idx
            menu_switch.inputs[idx + 1].hide = True
            menu_switch.outputs[idx + 1].hide = True

        menu_switch.hide = True
        menu_switch.location = _loc_format(loc)
        return menu_switch

    def create_IndexSwitch(self, typ, loc=(None, None)):
        idx_switch = IndexSwitch(len(self.enums), typ, loc)
        self >> idx_switch
        return idx_switch


class IndexSwitch(Node):
    def __init__(self, n: int, type: str = "INT", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeMenuSwitch" = self.__create(n, type, loc)
        self.type = type
        self.n = n
        self.orig_loc = loc

    def __create(self, n, typ, loc):
        idx_switch: "bpy.types.GeometryNodeMenuSwitch" = self.tree.nodes.new(
            "GeometryNodeIndexSwitch"
        )

        idx_switch.data_type = typ
        idx_switch.index_switch_items.clear()
        for _ in range(n):
            idx_switch.index_switch_items.new()

        idx_switch.location = _loc_format(loc)
        return idx_switch

    def __getitem__(self, item):
        return Socket(self, item + 1, is_output=False)


class Mix(Node):
    all_data_types = ("FLOAT", "VECTOR", "RGBA", "ROTATION")
    all_operations = {"MIX", "DARKEN", "MULTIPLY", "BURN", "LIGHTEN", "SCREEN", "DODGE", "ADD", "OVERLAY",
                      "SOFT_LIGHT", "LINEAR_LIGHT", "DIFFERENCE", "EXCLUSION", "SUBTRACT",
                      "DIVIDE", "HUE", "SATURATION", "COLOR", "VALUE",
    }  # fmt: skip

    def __init__(
        self, blend_type: str = "MIX", data_type="FLOAT", factor_mode="UNIFORM", loc=(None, None)
    ):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(
            blend_type, data_type, factor_mode, loc
        )
        self.blend_type = blend_type
        self.data_type = data_type
        self.factor_mode = factor_mode
        self.orig_loc = loc

    def __create(self, blend_type, data_type, factor_mode, loc):
        math_node = self.tree.nodes.new("ShaderNodeMix")
        math_node.blend_type = blend_type
        math_node.data_type = data_type
        math_node.factor_mode = factor_mode

        math_node.location = _loc_format(loc)
        return math_node

    @fproperty
    def Factor():  # noqa
        def getter(self):
            return Socket(self, int(self.factor_mode != "UNIFORM"), is_output=False)

        def setter(self, val: float | int):
            self.bnode.inputs[self.factor_mode != "UNIFORM"].default_value = val

        return getter, setter

    @fproperty
    def A():  # noqa
        def getter(self):
            for i, typ in zip(range(2, 9, 2), self.all_data_types):
                if self.data_type == typ:
                    return Socket(self, i, is_output=False)
            return self.__not_found_data_type()

        def setter(self, val: float | int):
            for i, typ in zip(range(2, 9, 2), self.all_data_types):
                if self.data_type == typ:
                    self.bnode.inputs[i].default_value = val
            else:
                self.__not_found_data_type()

        return getter, setter

    @fproperty
    def B():  # noqa
        def getter(self):
            for i, typ in zip(range(3, 10, 2), self.all_data_types):
                if self.data_type == typ:
                    return Socket(self, i, is_output=False)
            return self.__not_found_data_type()

        def setter(self, val: float | int):
            for i, typ in zip(range(3, 10, 2), self.all_data_types):
                if self.data_type == typ:
                    self.bnode.inputs[i].default_value = val
            else:
                self.__not_found_data_type()

        return getter, setter

    @property
    def Output(self):
        for i, typ in enumerate(self.all_data_types):
            if self.data_type == typ:
                return Socket(self, i, is_output=True)
        self.__not_found_data_type()
        return Socket(self, 4, is_output=True)

    def __not_found_data_type(self):
        raise NotImplementedError(f"UniV: Mix Mode: Not found {self.data_type!r} data type.")


class Math(Node):
    all_operations = {
        "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MULTIPLY_ADD", "POWER",
        "LOGARITHM", "SQRT", "INVERSE_SQRT", "ABSOLUTE", "EXPONENT", "MINIMUM",
        "MAXIMUM", "LESS_THAN", "GREATER_THAN", "SIGN", "COMPARE",
        "SMOOTH_MIN", "SMOOTH_MAX", "ROUND", "FLOOR", "CEIL", "TRUNC", "FRACT", "MODULO",
        "FLOORED_MODULO", "WRAP", "SNAP", "PINGPONG",
        "SINE", "COSINE", "TANGENT", "ARCSINE", "ARCCOSINE", "ARCTANGENT", "ARCTAN2",
        "SINH", "COSH", "TANH", "RADIANS", "DEGREES",
    }  # fmt: skip

    def __init__(self, operation: str = "ADD", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(operation, loc)
        self.operation = operation
        self.orig_loc = loc

    def __create(self, operation, loc):
        math_node = self.tree.nodes.new("ShaderNodeMath")
        math_node.operation = operation
        math_node.location = _loc_format(loc)
        return math_node

    @fproperty
    def A():  # noqa
        def getter(self):
            return Socket(self, 0, is_output=False)

        def setter(self, val: float | int):
            self.bnode.inputs[0].default_value = val

        return getter, setter

    @fproperty
    def B():  # noqa
        def getter(self):
            return Socket(self, 1, is_output=False)

        def setter(self, val: float | int):
            self.bnode.inputs[1].default_value = val

        return getter, setter

    @fproperty
    def C():  # noqa
        def getter(self):
            return Socket(self, 2, is_output=False)

        def setter(self, val: float | int):
            self.bnode.inputs[2].default_value = val

        return getter, setter

    @fproperty
    def Epsilon():  # noqa
        def getter(self):
            assert self.operation in {"LESS_THAN", "GREATER_THAN"}
            return Socket(self, 1, is_output=False)

        def setter(self, val: float | int):
            assert self.operation in {"LESS_THAN", "GREATER_THAN"}
            self.bnode.inputs[1].default_value = val

        return getter, setter


class BoundaryBox(Node):

    def __init__(self, loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(loc)
        self.orig_loc = loc

    def __create(self, loc):
        math_node = self.tree.nodes.new("GeometryNodeBoundBox")
        math_node.location = _loc_format(loc)
        return math_node

    @property
    def Max(self):
        return Socket(self, 1, is_output=True)

    @property
    def Min(self):
        return Socket(self, 2, is_output=True)

    # noinspection PyStatementEffect
    def get_lengths(self, loc=(None, None)):
        subtract = VectorMath("SUBTRACT")
        self.Max >> subtract.B  # noqa
        self.Min >> subtract.A  # noqa

        separate_xyz = SeparateXYZ(loc=loc)
        separate_xyz.label = "Lengths"
        separate_xyz.label = "Lengths XYZ"
        subtract >> separate_xyz
        return subtract, separate_xyz


class VectorMath(Math):
    all_operations = {
        "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MULTIPLY_ADD",
        "CROSS_PRODUCT", "PROJECT", "REFLECT", "REFRACT", "FACEFORWARD", "DOT_PRODUCT",
        "DISTANCE", "LENGTH", "SCALE", "NORMALIZE", "ABSOLUTE", "POWER", "SIGN", "MINIMUM", "MAXIMUM",
        "ROUND", "FLOOR", "CEIL", "FRACTION", "MODULO", "WRAP", "SNAP", "SINE", "COSINE", "TANGENT",
    }  # fmt: skip

    def __init__(self, operation: str = "ADD", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(operation, loc)
        self.operation = operation
        self.orig_loc = loc

    def __create(self, operation, loc):
        math_node: "bpy.types.GeometryNodeTree" = self.tree.nodes.new("ShaderNodeVectorMath")
        math_node.operation = operation
        math_node.location = _loc_format(loc)
        return math_node


class SeparateXYZ(SimpleNode):
    _node_name = "ShaderNodeSeparateXYZ"

    @property
    def X(self):
        return Socket(self, 0, is_output=True)

    @property
    def Y(self):
        return Socket(self, 1, is_output=True)

    @property
    def Z(self):
        return Socket(self, 2, is_output=True)

    def create_swizzled_CombineXYZ(
        self, swizzle: str, default_value=(0, 0, 0), hide=True, loc=(None, None)
    ):
        combine_node = CombineXYZ(default_value, loc)
        if hide:
            combine_node.bnode.hide = hide
            combine_node.bnode.width = 115  # Best width for avoid dots in label with zoom.
            self.bnode.hide = hide
            self.bnode.width = 50

        for comp_a, comp_b in zip("XYZ", swizzle):
            if comp_b != "-":
                getattr(self, comp_a) >> getattr(combine_node, comp_b)
        combine_node.label = f"Swizzle {swizzle.replace('-', '')}"
        return combine_node


class CombineXYZ(Node):
    def __init__(self, default_value=(0, 0, 0), loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(default_value, loc)
        self.orig_loc = loc

    def __create(self, default_value, loc):
        combine_node = self.tree.nodes.new("ShaderNodeCombineXYZ")
        for i, val in enumerate(default_value):
            combine_node.inputs[i].default_value = val

        combine_node.location = _loc_format(loc)
        return combine_node

    @property
    def X(self):
        return Socket(self, 0, is_output=False)

    @X.setter
    def X(self, val: float | int):
        self.bnode.inputs[0].default_value = val

    @property
    def Y(self):
        return Socket(self, 1, is_output=False)

    @Y.setter
    def Y(self, val: float | int):
        self.bnode.inputs[1].default_value = val

    @property
    def Z(self):
        return Socket(self, 2, is_output=False)

    @Z.setter
    def Z(self, val: float | int):
        self.bnode.inputs[2].default_value = val


class Position(SimpleNode):
    _node_name = "GeometryNodeInputPosition"


class SplitEdges(SimpleNode):
    _node_name = "GeometryNodeSplitEdges"

    @property
    def Selection(self):
        return Socket(self, 1, is_output=False)

class FaceArea(SimpleNode):
    _node_name = "GeometryNodeInputMeshFaceArea"

class MeshIsland(SimpleNode):
    _node_name = "GeometryNodeInputMeshIsland"

class FieldAverage(Node):
    def __init__(self, data_type="FLOAT_VECTOR", domain="POINT", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(data_type, domain, loc)
        self.orig_loc = loc

    def __create(self, data_type, domain, loc):
        avg_node = self.tree.nodes.new("GeometryNodeFieldAverage")
        avg_node.data_type = data_type
        avg_node.domain = domain
        avg_node.location = _loc_format(loc)
        return avg_node

    @property
    def GroupID(self):
        return Socket(self, 1, is_output=False)

class AccumulateField(Node):
    def __init__(self, data_type="FLOAT", domain="FACE", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(data_type, domain, loc)
        self.orig_loc = loc

    def __create(self, data_type, domain, loc):
        accum_node = self.tree.nodes.new("GeometryNodeAccumulateField")
        accum_node.data_type = data_type
        accum_node.domain = domain
        accum_node.location = _loc_format(loc)
        return accum_node

    @property
    def Total(self):
        return Socket(self, 2, is_output=True)

    @property
    def GroupID(self):
        return Socket(self, 1, is_output=False)

class SetPosition(SimpleNode):
    _node_name = "GeometryNodeSetPosition"

    @property
    def Selection(self):
        return Socket(self, 1, is_output=False)

    @property
    def Position(self):
        return Socket(self, 2, is_output=False)

    @property
    def Offset(self):
        return Socket(self, 3, is_output=False)

    @Offset.setter
    def Offset(self, val):
        self.bnode.inputs[3].default_value = val


class NamedAttribute(Node):
    def __init__(self, data_type="FLOAT_VECTOR", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(data_type, loc)
        self.orig_loc = loc

    def __create(self, data_type, loc):
        attribute = self.tree.nodes.new("GeometryNodeInputNamedAttribute")
        attribute.data_type = data_type
        attribute.location = _loc_format(loc)
        return attribute

    @property
    def Exists(self):
        return Socket(self, 1, is_output=True)

    @property
    def Name(self):
        return Socket(self, 0, is_output=False)

    @Name.setter
    def Name(self, val):
        self.bnode.inputs[0].default_value = val

class CaptureAttribute(Node):

    def __init__(self, domain = "POINT", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(domain, loc)
        self.orig_loc = loc

    def __create(self, domain, loc):
        attribute = self.tree.nodes.new("GeometryNodeCaptureAttribute")
        attribute.domain = domain
        attribute.capture_items.clear()
        attribute.location = _loc_format(loc)
        return attribute

    def add_vector(self, name):
        attr = self.bnode.capture_items.new("FLOAT", name)
        attr.data_type = "FLOAT_VECTOR"

    def add_float(self, name):
        attr = self.bnode.capture_items.new("FLOAT", name)
        attr.data_type = "FLOAT"

    def inputs(self, item):
        return Socket(self, item, is_output=False)

    def outputs(self, item):
        return Socket(self, item, is_output=True)


class ScaleElements(Node):

    def __init__(self, domain = "FACE", scale_mode="Uniform", axis=(1.0, 0.0, 0.0), loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(domain, scale_mode, axis, loc)
        self.orig_loc = loc

    def __create(self, domain, scale_mode, axis, loc):
        attribute = self.tree.nodes.new("GeometryNodeScaleElements")
        attribute.domain = domain
        attribute.inputs[4].default_value = scale_mode
        attribute.inputs[5].default_value = axis
        attribute.location = _loc_format(loc)
        return attribute

    @property
    def Scale(self):
        return Socket(self, 2, is_output=False)

    @property
    def Center(self):
        return Socket(self, 3, is_output=False)

class Switch(Node):
    all_input_type = {"FLOAT","INT","BOOLEAN","VECTOR","RGBA","ROTATION","MATRIX","STRING","MENU","OBJECT",
                  "IMAGE","GEOMETRY","COLLECTION","MATERIAL","BUNDLE","CLOSURE","FONT","SOUND",
    }  # fmt: skip

    def __init__(self, input_type="FLOAT", loc=(None, None)):  # noqa
        self.bnode: "bpy.types.GeometryNodeTree" = self.__create(input_type, loc)
        self.input_type = input_type
        self.orig_loc = loc

    def __create(self, input_type, loc):
        switch_node = self.tree.nodes.new("GeometryNodeSwitch")
        switch_node.input_type = input_type
        switch_node.location = _loc_format(loc)
        return switch_node

    @property
    def false(self):
        return Socket(self, 1, is_output=False)

    @false.setter
    def false(self, val):
        self.bnode.inputs[1].default_value = val

    @property
    def true(self):
        return Socket(self, 2, is_output=False)

    @true.setter
    def true(self, val):
        self.bnode.inputs[2].default_value = val


class GN_IsUVEdgeBoundary:
    name = "Is UV Edge Boundary"

    @classmethod
    def get(cls):
        for ng in reversed(bpy.data.node_groups):
            if ng.name.startswith(cls.name):
                if cls.is_changed(ng):
                    print(f"UniV: Flatten: Node Group {ng.name!r} is changed.")
                    if ng.users == 0:
                        bpy.data.node_groups.remove(ng)
                else:
                    return ng
        return cls.create()

    # noinspection PyStatementEffect
    @classmethod
    def create(cls):
        with gt(cls.name) as group:
            # Input
            input_group = Input(add_default_socket=False, loc=(-950, -120))
            input_group.add_str("Name", "UVMap")

            output_group = Output(add_default_socket=False, loc=(1050, 0))
            output_group.add_bool("Boolean", False, attribute_domain="POINT")

            uv_attr = NamedAttribute(
                loc=(-700, -85)
            )  # TODO: Add Attribute.attribute_domain (auto detect by name)

            reroute = Reroute(loc=(-30, -120))
            index = Index(loc=(-650, 165))

            # Node Corners of Edge
            crn_of_edge_01 = CornersOfEdge(0, loc=(-400, 220))
            crn_of_edge_02 = CornersOfEdge(1, loc=(-400, -170))

            # Node Offset Corner in Face
            next_crn_01 = crn_of_edge_01.next_in_face(loc=(-210, 80))
            next_crn_02 = crn_of_edge_02.next_in_face(loc=(-210, -280))

            # Evaluate Index
            eval_at_idx_01 = EvaluateAtIndex(loc=(40, 100))
            eval_at_idx_02 = EvaluateAtIndex(loc=(40, 290))
            eval_at_idx_03 = EvaluateAtIndex(loc=(40, -270))
            eval_at_idx_04 = EvaluateAtIndex(loc=(40, -100))

            # Boundary Compare
            cmp_01 = Compare(operation="NOT_EQUAL", loc=(400, 130))
            cmp_01.Threshold = 0.00001
            cmp_02 = Compare(operation="NOT_EQUAL", loc=(400, -100))
            cmp_02.Threshold = 0.00001

            bit_or_01 = BooleanMath(operation="OR", loc=(650, 0))
            bit_or_02 = BooleanMath(operation="OR", loc=(850, 0))

            uv_seam_attr = NamedAttribute("BOOLEAN", loc=(650, 150))
            uv_seam_attr.Name = "uv_seam"

            input_group >> uv_attr
            uv_seam_attr >> bit_or_02

            uv_attr >> reroute

            reroute >> eval_at_idx_01
            reroute >> eval_at_idx_02
            reroute >> eval_at_idx_03
            reroute >> eval_at_idx_04

            index >> crn_of_edge_01
            index >> crn_of_edge_02

            crn_of_edge_01 >> next_crn_01
            crn_of_edge_01 >> eval_at_idx_02.Index
            crn_of_edge_02 >> next_crn_02
            crn_of_edge_02 >> eval_at_idx_04.Index
            next_crn_01 >> eval_at_idx_01.Index
            next_crn_02 >> eval_at_idx_03.Index

            eval_at_idx_01 >> cmp_02.A
            eval_at_idx_02 >> cmp_01.A
            eval_at_idx_03 >> cmp_01.B
            eval_at_idx_04 >> cmp_02.B

            cmp_02 >> (bit_or_01, 1)
            cmp_01 >> bit_or_01
            bit_or_01 >> (bit_or_02, 1)
            bit_or_02 >> output_group

            return group

    @staticmethod
    def is_changed(group):
        name = GN_IsUVEdgeBoundary.__qualname__
        if not group:
            return True

        if len(group.nodes) != 18:
            print(f"UniV: {name!r} is changed, expected `18` nodes, got {len(group.nodes)}")
            return True

        sockets_count = sum(sk.is_linked for n in group.nodes for sk in n.inputs)
        if sockets_count != 23:
            print(f"UniV: {name!r} is changed, expected `23` linked sockets, got {sockets_count}")
            return True

        all_nodes_types = {
            "FIELD_AT_INDEX",
            "INDEX",
            "REROUTE",
            "INPUT_ATTRIBUTE",
            "OFFSET_CORNER_IN_FACE",
            "BOOLEAN_MATH",
            "GROUP_OUTPUT",
            "COMPARE",
            "GROUP_INPUT",
            "CORNERS_OF_EDGE",
        }
        if {n.type for n in group.nodes} != all_nodes_types:
            return True

        return False
