from keanu.vartypes import tensor_arg_types
from .base import Vertex
from .generated import ConstantBoolean, ConstantDouble, ConstantInteger
from keanu.infer_type import infer_type_and_execute


def Const(t: tensor_arg_types) -> Vertex:
    return infer_type_and_execute(t, {bool: ConstantBoolean, int: ConstantInteger, float: ConstantDouble})
