import bpy


class NewLinks:
    def __init__(self, group):
        self.group = group
        self.node_with_sk_idx = None

    def __call__(self, *node_with_sk_idx):
        if len(node_with_sk_idx) == 1:
            self.node_with_sk_idx = (node_with_sk_idx[0], 0)
        else:
            assert isinstance(node_with_sk_idx[1], int)
            self.node_with_sk_idx = node_with_sk_idx
        return self

    def __rshift__(self, node_input: tuple):
        input_idx = 0
        if isinstance(node_input, tuple):
            node_input, input_idx = node_input
            assert isinstance(input_idx, int)
        self.group.links.new(self.node_with_sk_idx[0].outputs[self.node_with_sk_idx[1]], node_input.inputs[input_idx])
        self.node_with_sk_idx = None


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

    @classmethod
    def create(cls):
        group = bpy.data.node_groups.new(type="GeometryNodeTree", name=cls.name)

        nodes = group.nodes

        # Input
        input_group = nodes.new("NodeGroupInput")
        input_group.location = (-950, -120)

        input_sk_uv = group.interface.new_socket(name="Name", in_out="INPUT", socket_type="NodeSocketString")
        input_sk_uv.default_value = "UVMap"
        input_sk_uv.subtype = "NONE"
        input_sk_uv.attribute_domain = "POINT"

        # Output
        output_group = nodes.new("NodeGroupOutput")
        output_group.location = (1050, 0)

        output_sk_cmp = group.interface.new_socket(name="Boolean", in_out="OUTPUT", socket_type="NodeSocketBool")
        output_sk_cmp.default_value = False
        output_sk_cmp.attribute_domain = "POINT"

        # UV Attribute
        uv_attr = nodes.new("GeometryNodeInputNamedAttribute")
        uv_attr.data_type = "FLOAT_VECTOR"
        uv_attr.location = (-700, -85)

        # Node Reroute
        reroute = nodes.new("NodeReroute")
        reroute.location = (-30, -120)

        # Node Index
        index = nodes.new("GeometryNodeInputIndex")
        index.location = (-650, 165)

        # Node Corners of Edge
        crn_of_edge_01 = nodes.new("GeometryNodeCornersOfEdge")
        crn_of_edge_01.inputs[2].default_value = 0  # Sort Index
        crn_of_edge_01.location = (-400, 220)

        crn_of_edge_02 = nodes.new("GeometryNodeCornersOfEdge")
        crn_of_edge_02.inputs[2].default_value = 1  # Sort Index
        crn_of_edge_02.location = (-400, -170)

        # Node Offset Corner in Face
        next_crn_01 = nodes.new("GeometryNodeOffsetCornerInFace")
        next_crn_01.inputs[1].default_value = 1  # Offset
        next_crn_01.location = (-210, 80)

        next_crn_02 = nodes.new("GeometryNodeOffsetCornerInFace")
        next_crn_02.inputs[1].default_value = 1  # Offset
        next_crn_02.location = (-210, -280)

        # Evaluate Index
        eval_at_idx_01 = nodes.new("GeometryNodeFieldAtIndex")
        eval_at_idx_01.data_type = "FLOAT_VECTOR"
        eval_at_idx_01.domain = "CORNER"
        eval_at_idx_01.location = (40, 100)

        eval_at_idx_02 = nodes.new("GeometryNodeFieldAtIndex")
        eval_at_idx_02.data_type = "FLOAT_VECTOR"
        eval_at_idx_02.domain = "CORNER"
        eval_at_idx_02.location = (40, 290)

        eval_at_idx_03 = nodes.new("GeometryNodeFieldAtIndex")
        eval_at_idx_03.data_type = "FLOAT_VECTOR"
        eval_at_idx_03.domain = "CORNER"
        eval_at_idx_03.location = (40, -270)

        eval_at_idx_04 = nodes.new("GeometryNodeFieldAtIndex")
        eval_at_idx_04.data_type = "FLOAT_VECTOR"
        eval_at_idx_04.domain = "CORNER"
        eval_at_idx_04.location = (40, -100)

        # Boundary Compare
        cmp_01 = nodes.new("FunctionNodeCompare")
        cmp_01.data_type = "VECTOR"
        cmp_01.mode = "ELEMENT"
        cmp_01.operation = "NOT_EQUAL"
        cmp_01.inputs[12].default_value = 0.00001  # Epsilon
        cmp_01.location = (400, 130)

        cmp_02 = nodes.new("FunctionNodeCompare")
        cmp_02.data_type = "VECTOR"
        cmp_02.mode = "ELEMENT"
        cmp_02.operation = "NOT_EQUAL"
        cmp_02.inputs[12].default_value = 0.00001  # Epsilon
        cmp_02.location = (400, -100)

        bit_or_01 = nodes.new("FunctionNodeBooleanMath")
        bit_or_01.operation = "OR"
        bit_or_01.location = (650, 0)

        # Seam Compare
        uv_seam_attr = nodes.new("GeometryNodeInputNamedAttribute")
        uv_seam_attr.data_type = "BOOLEAN"
        uv_seam_attr.inputs[0].default_value = "uv_seam"  # Name
        uv_seam_attr.location = (650, 150)

        bit_or_02 = nodes.new("FunctionNodeBooleanMath")
        bit_or_02.operation = "OR"
        bit_or_02.location = (850, 0)


        # Initialize links
        new_links = NewLinks(group)

        new_links(input_group) >> uv_attr  # Name > Name
        new_links(uv_seam_attr) >> bit_or_02  # Attribute > Boolean
        # Edge Boundary > Seam Boundary
        new_links(uv_attr) >> reroute  # Attribute > Reroute

        # Reroute > Evaluate At Index
        new_links(reroute) >> eval_at_idx_01
        new_links(reroute) >> eval_at_idx_02
        new_links(reroute) >> eval_at_idx_03
        new_links(reroute) >> eval_at_idx_04

        new_links(index) >> crn_of_edge_01  # Index > Edge Index
        new_links(index) >> crn_of_edge_02  # Index > Edge Index

        new_links(crn_of_edge_01) >> next_crn_01  # Corner Index > Corner Index
        new_links(crn_of_edge_01) >> (eval_at_idx_02, 1)  # Corner Index > Index
        new_links(crn_of_edge_02) >> next_crn_02  # Corner Index > Corner Index
        new_links(crn_of_edge_02) >> (eval_at_idx_04, 1)  # Corner Index > Index
        new_links(next_crn_01) >> (eval_at_idx_01, 1)  # Corner Index > Index
        new_links(next_crn_02) >> (eval_at_idx_03, 1)  # Corner Index > Index

        new_links(eval_at_idx_01) >> (cmp_02, 4)  # Value > A
        new_links(eval_at_idx_02) >> (cmp_01, 4)  # Value > A
        new_links(eval_at_idx_03) >> (cmp_01, 5)  # Value > B
        new_links(eval_at_idx_04) >> (cmp_02, 5)  # Value > B

        new_links(cmp_02) >> (bit_or_01, 1)  # Result > Boolean
        new_links(cmp_01) >> bit_or_01  # Result > Boolean
        new_links(bit_or_01) >> (bit_or_02, 1)  # A > B
        new_links(bit_or_02) >> output_group  # Boolean > Boolean

        return group

    @staticmethod
    def is_changed(group):
        if not group:
            return True

        if len(group.nodes) != 18:
            return True

        sockets_count = sum(sk.is_linked for n in group.nodes for sk in n.inputs)
        if sockets_count != 23:
            return True

        all_nodes_types = {'FIELD_AT_INDEX', 'INDEX', 'REROUTE', 'INPUT_ATTRIBUTE', 'OFFSET_CORNER_IN_FACE',
                           'BOOLEAN_MATH', 'GROUP_OUTPUT', 'COMPARE', 'GROUP_INPUT', 'CORNERS_OF_EDGE'}
        if {n.type for n in group.nodes} != all_nodes_types:
            return True

        return False