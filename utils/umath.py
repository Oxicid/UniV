import math
from bl_math import lerp
from mathutils import Vector


def vec_isclose(a, b, abs_tol: float = 0.00001):
    return all(math.isclose(a1, b1, abs_tol=abs_tol) for a1, b1 in zip(a, b))

def vec_isclose_to_uniform(delta: Vector, abs_tol: float = 0.00001):
    return math.isclose(delta.x, 1.0, abs_tol=abs_tol) and math.isclose(delta.y, 1.0, abs_tol=abs_tol)

def vec_isclose_to_zero(delta: Vector, abs_tol: float = 0.00001):
    return math.isclose(delta.x, 0, abs_tol=abs_tol) and math.isclose(delta.y, 0, abs_tol=abs_tol)


# https://gist.github.com/laundmo/b224b1f4c8ef6ca5fe47e132c8deab56
def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linar Interpolation, get the fraction between a and b on which v resides.
    Examples
    --------
        0.5 == inv_lerp(0, 100, 50)
        0.8 == inv_lerp(1, 5, 4.2)
    """
    return (v - a) / (b - a)

def remap(i_min: float, i_max: float, o_min: float, o_max: float, v: float) -> float:
    """Remap values from one linear scale to another, a combination of lerp and inv_lerp.
    i_min and i_max are the scale on which the original value resides,
    o_min and o_max are the scale to which it should be mapped.
    Examples
    --------
        45 == remap(0, 100, 40, 50, 50)
        6.2 == remap(1, 5, 3, 7, 4.2)
    """
    return lerp(o_min, o_max, inv_lerp(i_min, i_max, v))
