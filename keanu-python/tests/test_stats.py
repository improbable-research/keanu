import pytest
from keanu import stats
import numpy as np

def test_can_get_autocorrelation():
    x = np.arange(1,9)
    x_list = [np.array(a) for a in x]
    autocorr = stats.autocorrelation(x_list)
    expected = [1., 0.625, 0.27380952, -0.0297619, -0.26190476,
            -0.39880952, -0.41666667, -0.29166667]
    np.testing.assert_almost_equal(autocorr,expected)
