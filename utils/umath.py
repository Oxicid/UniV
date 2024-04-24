import math
from mathutils import Vector


def vec_isclose_to_uniform(delta: Vector, abs_tol: float = 0.00001):
    return math.isclose(delta.x, 1.0, abs_tol=abs_tol) and math.isclose(delta.y, 1.0, abs_tol=abs_tol)

def vec_isclose_to_zero(delta: Vector, abs_tol: float = 0.00001):
    return math.isclose(delta.x, 0, abs_tol=abs_tol) and math.isclose(delta.y, 0, abs_tol=abs_tol)
