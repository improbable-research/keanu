from typing import Iterable, Union, Type, Any, TYPE_CHECKING
import numpy as np
from pandas import Series, DataFrame
from keanu.base import JavaObjectWrapper
#from keanu.vertex.base import VertexOps

int_types = Union[int, np.integer]
float_types = Union[int, np.float]
bool_types = Union[bool, np.bool_]

primitive_types = Union[int_types, float_types, bool_types]

pandas_types = Union[Series, DataFrame]
numpy_types = Union[np.ndarray]

const_arg_types = Union[primitive_types, pandas_types, numpy_types]
vertex_arg_types = Union[const_arg_types, JavaObjectWrapper]
shape_types = Iterable[primitive_types]

runtime_numpy_types : Type[Any]
runtime_pandas_types : Type[Any]
runtime_primitive_types : Type[Any]
runtime_int_types : Type[Any]
runtime_float_types : Type[Any]
runtime_bool_types : Type[Any]
runtime_const_arg_types : Type[Any]

if not TYPE_CHECKING:
    runtime_numpy_types = numpy_types
    runtime_pandas_types = pandas_types.__args__
    runtime_primitive_types = primitive_types.__args__
    runtime_int_types = int_types.__args__
    runtime_float_types = float_types.__args__
    runtime_bool_types = bool_types.__args__
    runtime_const_arg_types = const_arg_types.__args__
