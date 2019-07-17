from typing import Union

from keanu.context import KeanuContext
from keanu.infer_type import get_type_of_value
from keanu.vartypes import tensor_arg_types
from .base import Vertex, Double, Integer, Boolean
from .generated import cast_to_boolean_vertex, cast_to_double_vertex, cast_to_integer_vertex

context = KeanuContext()


def If(predicate: Union[tensor_arg_types, Vertex], thn: Union[tensor_arg_types, Vertex],
       els: Union[tensor_arg_types, Vertex]) -> Vertex:
    then_type = get_type_of_value(thn)
    else_type = get_type_of_value(els)

    if then_type == float or else_type == float:
        return Double(context.jvm_view().IfVertex, None, cast_to_boolean_vertex(predicate), cast_to_double_vertex(thn),
                      cast_to_double_vertex(els))
    elif then_type == int or else_type == int:
        return Integer(context.jvm_view().IfVertex, None, cast_to_boolean_vertex(predicate),
                       cast_to_integer_vertex(thn), cast_to_integer_vertex(els))
    elif then_type == bool and else_type == bool:
        return Boolean(context.jvm_view().IfVertex, None, cast_to_boolean_vertex(predicate),
                       cast_to_boolean_vertex(thn), cast_to_boolean_vertex(els))
    else:
        raise NotImplementedError("Unexpected types for If statement: {}, {}".format(then_type, else_type))
