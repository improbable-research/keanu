import numpy as np
import pandas as pd
from typing import Any

int_types = (int, np.integer)

float_types = (float, np.float)

bool_types = (bool, np.bool_)

primitive_types = int_types + float_types + bool_types

pandas_types = (pd.Series, pd.DataFrame)

numpy_types : Any = (np.ndarray, )

const_arg_types : Any = numpy_types + pandas_types + primitive_types
