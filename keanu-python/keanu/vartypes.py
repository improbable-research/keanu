from typing import Iterable, Union, Type, Any, TYPE_CHECKING
import numpy as np
from pandas import Series, DataFrame
from keanu.vertex.ops import VertexOps

# Primitive types
int_types = Union[int, np.integer]
float_types = Union[float, np.float16, np.float32, np.float64]
bool_types = Union[bool, np.bool_]

primitive_types = Union[int_types, float_types, bool_types, np.generic]

# Tensor arg types
# some static types are merged because mypy thinks bool is a subtype of int and int is a subtype of float
# see : https://github.com/python/mypy/issues/1850
pandas_types = Union[Series, DataFrame]
numpy_types = Union[np.ndarray]
pandas_and_numpy_types = Union[pandas_types, numpy_types]

tensor_arg_types = Union[int_types, bool_types, float_types, pandas_and_numpy_types]
bool_tensor_arg_types = Union[bool_types, pandas_and_numpy_types]
int_and_bool_tensor_arg_types = Union[int_types, bool_types, pandas_and_numpy_types]

# Vertex arg types
# some static types are merged because mypy thinks bool is a subtype of int and int is a subtype of float
# see : https://github.com/python/mypy/issues/1850
const_arg_types = Union[primitive_types, tensor_arg_types]

vertex_operable_types = Union[VertexOps]
vertex_arg_types = Union[tensor_arg_types, vertex_operable_types]
bool_vertex_arg_types = Union[bool_tensor_arg_types, vertex_operable_types]
int_and_bool_vertex_arg_types = Union[int_and_bool_tensor_arg_types, vertex_operable_types]

# Shape types
shape_types = Iterable[primitive_types]

# mypy prohibits use of types from typing module at runtime
# see : https://github.com/python/mypy/issues/5354
runtime_numpy_types : Type[Any]
runtime_pandas_types : Type[Any]
runtime_primitive_types : Type[Any]
runtime_int_types : Type[Any]
runtime_float_types : Type[Any]
runtime_bool_types : Type[Any]
runtime_const_arg_types : Type[Any]
runtime_vertex_operable_types : Type[Any]

if not TYPE_CHECKING:
    # Union with one argument does not have __args__ parameter
    runtime_numpy_types = numpy_types 
    runtime_pandas_types = pandas_types.__args__
    runtime_primitive_types = primitive_types.__args__
    runtime_int_types = int_types.__args__
    runtime_float_types = float_types.__args__
    runtime_bool_types = bool_types.__args__
    runtime_const_arg_types = const_arg_types.__args__
    runtime_vertex_operable_types  = vertex_operable_types
