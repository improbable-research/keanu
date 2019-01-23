from keanu.infer_type import infer_type_and_execute, get_type_of_value
from keanu.vartypes import tensor_arg_types
from keanu.vertex import do_inferred_vertex_cast, Boolean
from .base import Vertex, vertex_constructor_param_types
from .generated import BooleanIf, DoubleIf, IntegerIf, ConstantBoolean, ConstantInteger, ConstantDouble


def __cast_to_vertex(input: vertex_constructor_param_types) -> Vertex:
    return do_inferred_vertex_cast({bool: ConstantBoolean, int: ConstantInteger, float: ConstantDouble}, input)


def If(predicate: Vertex, thn: Vertex, els: Vertex) -> Vertex:
    predicate = __cast_to_vertex(predicate)
    thn = __cast_to_vertex(thn)
    els = __cast_to_vertex(els)

    if type(predicate) != Boolean:
        raise TypeError("Predicate must be boolean: got {}".format(type(predicate)))

    then_type = get_type_of_value(thn.get_value())
    else_type = get_type_of_value(els.get_value())
    if (then_type != else_type):
        raise TypeError('The "then" and "else" clauses must be of the same datatype: {} vs {}'.format(then_type, else_type))

    if then_type == bool:
        return BooleanIf(predicate, thn, els)
    elif then_type == int:
        return IntegerIf(predicate, thn, els)
    elif then_type == float:
        return DoubleIf(predicate, thn, els)
    else:
        raise NotImplementedError("Unexpected type {} for vertex {}".format(then_type, thn))
