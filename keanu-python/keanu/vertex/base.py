import numpy as np
import collections
import numbers

import keanu as kn
from keanu.context import KeanuContext
from keanu.base import JavaObjectWrapper
from keanu.tensor import Tensor
from keanu.vartypes import primitive_types, const_arg_types, numpy_types, runtime_const_arg_types, runtime_primitive_types
from .ops import VertexOps
from typing import List, Any, Tuple, Iterator


k = KeanuContext()


class Vertex(JavaObjectWrapper, VertexOps):
    def __init__(self, val : Any, *args : Any) -> None:
        if args:
            ctor = val
            val = ctor(*(Vertex.__parse_args(args)))

        super(Vertex, self).__init__(val)

    def __hash__(self) -> int:
        return hash(self.get_id())

    def observe(self, v : const_arg_types) -> None:
        self.unwrap().observe(Tensor(v).unwrap())

    def set_value(self, v : const_arg_types) -> None:
        self.unwrap().setValue(Tensor(v).unwrap())

    def set_and_cascade(self, v : const_arg_types) -> None:
        self.unwrap().setAndCascade(Tensor(v).unwrap())

    def sample(self) -> numpy_types:
        return Tensor._to_ndarray(self.unwrap().sample())

    def get_value(self) -> numpy_types:
        return Tensor._to_ndarray(self.unwrap().getValue())

    def get_connected_graph(self) -> Iterator['Vertex']:
        return Vertex._to_generator(self.unwrap().getConnectedGraph())

    def get_id(self) -> Tuple[Any, ...]:
        return Vertex._get_python_id(self.unwrap())

    @staticmethod
    def __parse_args(args : Tuple[Any, ...]) -> List[Any]:
        return list(map(Vertex.__parse_arg, args))

    @staticmethod
    def __parse_arg(arg : Any) -> Any:
        if isinstance(arg, runtime_const_arg_types):
            return kn.vertex.const.Const(arg).unwrap() # type: ignore
        elif isinstance(arg, JavaObjectWrapper):
            return arg.unwrap()
        elif isinstance(arg, collections.Iterable) and all(isinstance(x, runtime_primitive_types) for x in arg):
            return k.to_java_long_array(arg)
        else:
            raise ValueError("Can't parse generic argument. Was given {}".format(type(arg)))

    @staticmethod
    def _to_generator(java_vertices : Any) -> Iterator['Vertex']:
        return (Vertex(java_vertex) for java_vertex in java_vertices)

    @staticmethod
    def _get_python_id(java_vertex : Any) -> Tuple[Any, ...]:
        return tuple(java_vertex.getId().getValue())
