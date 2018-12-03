from py4j.java_gateway import java_import
from .context import KeanuContext
from numpy import ndarray, fromiter
import numpy as np
from typing import List

k = KeanuContext()

java_import(k.jvm_view(), "io.improbable.keanu.algorithms.SampleStats")

def autocorrelation(arg: List[ndarray]) -> ndarray:
    arg_array = np.hstack(arg).astype(float)
    if arg_array.ndim != 1:
        raise ValueError("Autocorrelation must be run on a list of dimension 1 ndarrays.")
    autocorr = k.jvm_view().SampleStats.autocorrelation(k.to_java_array(arg_array))
    return fromiter(autocorr, float)