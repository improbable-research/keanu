from typing import List

from numpy import ndarray, fromiter, issubdtype, hstack, floating
from py4j.java_gateway import java_import

from keanu.vartypes import numpy_types
from .context import KeanuContext

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.SampleStats")


def autocorrelation(arg: List[numpy_types]) -> ndarray:
    all_floats = all(issubdtype(elem.dtype, floating) for elem in arg)
    if not all_floats:
        raise ValueError('Autocorrelation must be run on a list of numpy floating types.')
    all_shape_scalar = all(elem.shape == () for elem in arg)
    if not all_shape_scalar:
        raise ValueError("Autocorrelation must be run on a list of single element ndarrays.")
    arg_array = hstack(arg)
    autocorr = k.jvm_view().SampleStats.autocorrelation(k.to_java_array(arg_array))
    return fromiter(autocorr, arg[0].dtype)
