import numpy as np
import pandas as pd
from typing import Any, Iterable, Union, Tuple
from keanu.base import JavaObjectWrapper
from keanu.vertex.base import VertexOps

int_types = (int, np.integer)

float_types = (float, np.float_)

bool_types = (bool, np.bool_)

primitive_types = int_types + float_types + bool_types

pandas_types = (pd.Series, pd.DataFrame)

numpy_types = (np.ndarray, )

const_arg_types = (int, float, bool, np.integer, np.float_, np.bool_, pd.Series, pd.DataFrame, np.ndarray)#numpy_types + pandas_types + primitive_types

mypy_shape_types = Iterable[Union[int, float, bool, np.integer, np.float_, np.bool_]]

mypy_const_arg_types = Union[int, float, bool, np.integer, np.float_, np.bool_, pd.Series, pd.DataFrame, np.ndarray]

mypy_vertex_arg_types = Union[int, float, bool, np.integer, np.float_, np.bool_, pd.Series, pd.DataFrame, np.ndarray, JavaObjectWrapper, VertexOps]
