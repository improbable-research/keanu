from keanu.vartypes import tensor_arg_types
from .base import Vertex
from .generated import ConstantBoolean, ConstantDouble, ConstantInteger
from keanu.infer_type import infer_type_and_execute
from typing import Optional


def Const(t: tensor_arg_types, label: Optional[str] = None) -> Vertex:
    vertex = infer_type_and_execute(t, {bool: ConstantBoolean, int: ConstantInteger, float: ConstantDouble})
    if label is not None:
        vertex.set_label(label)
    return vertex
