from keanu.infer_type import infer_type_and_execute, get_type_of_value
from keanu.vartypes import tensor_arg_types
from keanu.vertex import do_inferred_vertex_cast
from .base import Vertex, vertex_constructor_param_types
from .generated import BooleanIf, DoubleIf, IntegerIf, ConstantBoolean, ConstantInteger, ConstantDouble


def __cast_to_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_inferred_vertex_cast({bool: ConstantBoolean, int: ConstantInteger, float: ConstantDouble}, input)


def If(predicate: Vertex, thn: Vertex, els: Vertex) -> Vertex:
    predicate = __cast_to_vertex(predicate)
    thn = __cast_to_vertex(thn)
    els = __cast_to_vertex(els)

    type_ = get_type_of_value(thn.get_value())

    if type_ == bool:
        return BooleanIf(predicate, thn, els)
    elif type_ == int:
        return IntegerIf(predicate, thn, els)
    elif type_ == float:
        return DoubleIf(predicate, thn, els)
    else:
        raise NotImplementedError("Unexpected type {} for vertex {}".format(type_, thn))
