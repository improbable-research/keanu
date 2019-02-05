from keanu.vertex import Gaussian, Uniform
import numpy as np


def test_tensor_example_creation():
    # %%SNIPPET_START%% PythonVertexFromNDArray
    mu = np.array([[2., 3., 4.],
                   [1., 4., 7.]])
    sigma = np.ones([2, 3])
    g = Gaussian(mu, sigma)
    # %%SNIPPET_END%% PythonVertexFromNDArray
