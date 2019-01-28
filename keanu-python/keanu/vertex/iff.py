from typing import Union

from keanu.infer_type import get_type_of_value
from keanu.vartypes import tensor_arg_types
from .base import Vertex
from .generated import BooleanIf, DoubleIf, IntegerIf


def If(predicate: Union[tensor_arg_types, Vertex], thn: Union[tensor_arg_types, Vertex],
       els: Union[tensor_arg_types, Vertex]) -> Vertex:

    predicate_type = get_type_of_value(predicate)
    then_type = get_type_of_value(thn)
    else_type = get_type_of_value(els)

    if predicate_type != bool:
        raise TypeError("Predicate must be boolean: got {}".format(type(predicate)))

    if (then_type != else_type):
        raise TypeError('The "then" and "else" clauses must be of the same datatype: {} vs {}'.format(
            then_type, else_type))

    if then_type == bool:
        return BooleanIf(predicate, thn, els)
    elif then_type == int:
        return IntegerIf(predicate, thn, els)
    elif then_type == float:
        return DoubleIf(predicate, thn, els)
    else:
        raise NotImplementedError("Unexpected type {} for vertex {}".format(then_type, thn))
