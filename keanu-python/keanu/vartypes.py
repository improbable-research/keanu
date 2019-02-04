from typing import Iterable, Union, Type, Any, TYPE_CHECKING, Dict, Tuple, List, Generator
from numpy import integer, floating, bool_, ndarray
from pandas import Series, DataFrame
from .base import JavaObjectWrapper

# see numpy's scalar hierarchy: https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#scalars
int_types = Union[int, integer]
float_types = Union[float, floating]
bool_types = Union[bool, bool_]
str_types = Union[str]

primitive_types = Union[int_types, float_types, bool_types]
'''
Tensor arg types
'''
pandas_types = Union[Series, DataFrame]
numpy_types = Union[ndarray]

tensor_arg_types = Union[primitive_types, pandas_types, numpy_types]
'''
Vertex arg types
'''
wrapped_java_types = Union[JavaObjectWrapper]
shape_types = Iterable[primitive_types]
'''
Sample types
'''
sample_types = Dict[Union[str, Tuple[str, str]], List[primitive_types]]
sample_generator_dict_type = Dict[Union[str, Tuple[str, str]], primitive_types]
sample_generator_types = Generator[Dict[Union[str, Tuple[str, str]], primitive_types], None, None]
'''
Runtime types
'''
# mypy prohibits use of static types from typing module at runtime
# see : https://github.com/python/mypy/issues/5354
runtime_numpy_types: Type[Any]
runtime_pandas_types: Type[Any]
runtime_primitive_types: Type[Any]
runtime_int_types: Type[Any]
runtime_float_types: Type[Any]
runtime_bool_types: Type[Any]
runtime_str_types: Type[Any]
runtime_tensor_arg_types: Type[Any]
runtime_wrapped_java_types: Type[Any]

if not TYPE_CHECKING:
    # Unions with a single element can be directly used as runtime type
    runtime_numpy_types = numpy_types
    runtime_pandas_types = pandas_types.__args__
    runtime_primitive_types = primitive_types.__args__
    runtime_int_types = int_types.__args__
    runtime_float_types = float_types.__args__
    runtime_bool_types = bool_types.__args__
    runtime_str_types = str_types
    runtime_tensor_arg_types = tensor_arg_types.__args__
    runtime_wrapped_java_types = wrapped_java_types
