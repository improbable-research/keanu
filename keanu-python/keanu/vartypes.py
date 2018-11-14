from typing import Iterable, Union, Type, Any, TYPE_CHECKING
from numpy import integer, floating, bool_, ndarray
from pandas import Series, DataFrame
from keanu.vertex.ops import VertexOps
from .base import JavaObjectWrapper


# see numpy's scalar hierarchy: https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#scalars
int_types = Union[int, integer]
float_types = Union[float, floating]
bool_types = Union[bool, bool_]

primitive_types = Union[int_types, float_types, bool_types]

'''
Tensor arg types
'''
pandas_types = Union[Series, DataFrame]
numpy_types = Union[ndarray]
pandas_and_numpy_types = Union[pandas_types, numpy_types]

# mypy treats bool as a subtype of int and int as a subtype of float
# see : https://github.com/python/mypy/issues/1850
tensor_arg_types = Union[primitive_types, pandas_and_numpy_types]
bool_tensor_arg_types = Union[bool_types, pandas_and_numpy_types]
int_and_bool_tensor_arg_types = Union[int_types, bool_types, pandas_and_numpy_types]

'''
Vertex arg types
'''
wrapped_java_types = Union[JavaObjectWrapper]
shape_types = Iterable[primitive_types]

# mypy treats bool as a subtype of int and int as a subtype of float
# see : https://github.com/python/mypy/issues/1850
vertex_param_types = Union[tensor_arg_types, wrapped_java_types, VertexOps]
bool_vertex_param_types = Union[bool_tensor_arg_types, wrapped_java_types, VertexOps]
int_and_bool_vertex_param_types = Union[int_and_bool_tensor_arg_types, wrapped_java_types, VertexOps]

vertex_arg_types = Union[vertex_param_types, shape_types]

'''
Runtime types
'''
# mypy prohibits use of static types from typing module at runtime
# see : https://github.com/python/mypy/issues/5354
runtime_numpy_types : Type[Any]
runtime_pandas_types : Type[Any]
runtime_primitive_types : Type[Any]
runtime_int_types : Type[Any]
runtime_float_types : Type[Any]
runtime_bool_types : Type[Any]
runtime_tensor_arg_types : Type[Any]
runtime_wrapped_java_types : Type[Any]

if not TYPE_CHECKING:
    # Union with one argument does not have __args__ parameter
    runtime_numpy_types = numpy_types
    runtime_pandas_types = pandas_types.__args__
    runtime_primitive_types = primitive_types.__args__
    runtime_int_types = int_types.__args__
    runtime_float_types = float_types.__args__
    runtime_bool_types = bool_types.__args__
    runtime_tensor_arg_types = tensor_arg_types.__args__
    runtime_wrapped_java_types = wrapped_java_types
