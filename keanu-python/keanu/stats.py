from typing import List, Tuple

from numpy import ndarray, fromiter, issubdtype, floating, stack
from py4j.java_gateway import java_import

from keanu.shape_validation import check_index_is_valid, check_all_shapes_match
from keanu.vartypes import numpy_types
from .context import KeanuContext

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.statistics.Autocorrelation")


def autocorrelation(arg: List[numpy_types], index: Tuple[int, ...] = ()) -> ndarray:
    check_all_floats(arg)
    check_all_shapes_match([elem.shape for elem in arg])
    check_index_is_valid(arg[0].shape, index)
    arg_array = stack(arg, axis=-1)[index]
    autocorr = k.jvm_view().Autocorrelation.calculate(k.to_java_array(arg_array))
    return fromiter(autocorr, arg[0].dtype)


def check_all_floats(arg: List[numpy_types]) -> None:
    all_floats = all(issubdtype(elem.dtype, floating) for elem in arg)
    if not all_floats:
        raise ValueError("Autocorrelation must be run on a list of numpy floating types.")
