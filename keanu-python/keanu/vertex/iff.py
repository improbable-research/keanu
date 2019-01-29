from typing import Union

from keanu.infer_type import get_type_of_value
from keanu.vartypes import tensor_arg_types
from .base import Vertex
from .generated import BooleanIf, DoubleIf, IntegerIf


def If(predicate: Union[tensor_arg_types, Vertex], thn: Union[tensor_arg_types, Vertex],
       els: Union[tensor_arg_types, Vertex]) -> Vertex:

    then_type = get_type_of_value(thn)
    else_type = get_type_of_value(els)

    if then_type == float or else_type == float:
        return DoubleIf(predicate, thn, els)
    elif then_type == int or else_type == int:
        return IntegerIf(predicate, thn, els)
    elif then_type == bool and else_type == bool:
        return BooleanIf(predicate, thn, els)
    else:
        raise NotImplementedError("Unexpected types for If statement: {}, {}".format(then_type, else_type))
