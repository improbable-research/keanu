import keanu as kn
import numpy as np
import sys
import logging


def test_appveyor_hanging():
    print("test_appveyor_hanging")
    logging.warn("test_appveyor_hanging")
    sys.stdout.flush()
    values = np.array([[1., 1.]])
    print("values", values)
    logging.warn("values %s" % values)
    sys.stdout.flush()
    const = kn.Const(values)
    print("const vertex shape", [i for i in const.getShape()])
    logging.warn("const vertex shape %s" % [i for i in const.getShape()])
    print("const vertex value", [i for i in const.getValue().asFlatArray()])
    logging.warn("const vertex value %s" % [i for i in const.getValue().asFlatArray()])
    sys.stdout.flush()
    exponential = kn.Exponential(const)


if __name__ == "__main__":
    test_appveyor_hanging()