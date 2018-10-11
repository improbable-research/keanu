import keanu as kn
import numpy as np


def test_appveyor_hanging():
    print("test_appveyor_hanging")
    values = np.array([[1., 1.]])
    print("values", values)
    const = kn.Const(values)
    print("const vertex shape", [i for i in const.getShape()])
    print("const vertex value", [i for i in const.getValue().asFlatArray()])
    exponential = kn.Exponential(const)


if __name__ == "__main__":
    test_appveyor_hanging()