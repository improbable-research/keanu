from keanu.infer_type import infer_type_and_execute, get_type_of_value
from keanu.vartypes import tensor_arg_types
from .base import Vertex
from .generated import BooleanIf, DoubleIf, IntegerIf


def If(predicate: Vertex, thn: Vertex, els: Vertex) -> Vertex:
    type_ = get_type_of_value(thn.get_value())

    if type_ == bool:
        return BooleanIf(predicate, thn, els)
    elif type_ == int:
        return IntegerIf(predicate, thn, els)
    elif type_ == float:
        return DoubleIf(predicate, thn, els)
    else:
        raise NotImplementedError("Unexpected type {} for vertex {}".format(type_, thn))
