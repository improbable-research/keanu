import keanu as kn
import numpy as np


def test_appveyor_freezing():
    values = np.array([[1., 1.]])
    const = kn.Const(values)
    exponential = kn.Exponential(const)
