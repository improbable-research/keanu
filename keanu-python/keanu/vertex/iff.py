from typing import Union

import keanu as kn
from keanu.infer_type import get_type_of_value
from keanu.vartypes import tensor_arg_types
from .base import Vertex


def Iff(predicate: Union[tensor_arg_types, Vertex], thn: Union[tensor_arg_types, Vertex],
        els: Union[tensor_arg_types, Vertex]) -> Vertex:
    then_type = get_type_of_value(thn)
    else_type = get_type_of_value(els)

    if then_type == float or else_type == float:
        return kn.vertex.generated.If(predicate, thn, els)
    elif then_type == int or else_type == int:
        return kn.vertex.generated.If(predicate, thn, els)
    elif then_type == bool and else_type == bool:
        return kn.vertex.generated.If(predicate, thn, els)
    else:
        raise NotImplementedError("Unexpected types for If statement: {}, {}".format(then_type, else_type))
