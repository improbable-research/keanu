from typing import List

import numpy as np
from numpy import ndarray, fromiter
from py4j.java_gateway import java_import

from keanu.vartypes import primitive_types
from .context import KeanuContext

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.statistics.Autocorrelation")


def autocorrelation(arg: List[primitive_types]) -> ndarray:
    check_all_floats(arg)
    autocorr = k.jvm_view().Autocorrelation.calculate(k.to_java_array(arg))
    return fromiter(autocorr, float)


def check_all_floats(arg: List[primitive_types]) -> None:
    all_floats = all(
        (type(elem) == float or type(elem) == np.float16 or type(elem) == np.float32 or type(elem) == np.float64)
        for elem in arg)
    if not all_floats:
        raise ValueError("Autocorrelation must be run on a list of floating types.")
