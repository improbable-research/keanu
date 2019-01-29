from typing import List, Tuple, Any
from numpy import ndarray, fromiter, issubdtype, floating, stack
from py4j.java_gateway import java_import

from keanu.shape_validation import check_index_is_valid, check_all_shapes_match
from keanu.vartypes import numpy_types
from .context import KeanuContext

import numpy as np

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.statistics.Autocorrelation")


def autocorrelation(arg: List[Any]) -> ndarray:
    check_all_floats(arg)
    autocorr = k.jvm_view().Autocorrelation.calculate(k.to_java_array(arg))
    return fromiter(autocorr, float)


def check_all_floats(arg: List[Any]) -> None:
    print(type(arg[0]))
    all_floats = all((type(elem) == float or type(elem) == np.float64) for elem in arg)
    if not all_floats:
        raise ValueError("Autocorrelation must be run on a list of floating types.")
