package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorMultivariateGaussian {

    private TensorMultivariateGaussian() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor covariance, KeanuRandom random) {
        DoubleTensor choleskyDecompOfCovariance = covariance.choleskyDecomposition();
        GaussianVertex gaussianVariates = new GaussianVertex(mu.getShape(), 0, 1);
        DoubleTensor samplesFromGaussianVariates = gaussianVariates.sample(random);
        return mu.plus(choleskyDecompOfCovariance.matrixMultiply(samplesFromGaussianVariates));
    }

}
