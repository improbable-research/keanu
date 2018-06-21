package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.distributions.continuous.Gaussian;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorMultivariateGaussian {

    private TensorMultivariateGaussian() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor covariance, KeanuRandom random) {
        DoubleTensor choleskyDecomp = covariance.choleskyDecomposition();
        GaussianVertex gaussian = new GaussianVertex(new int[]{mu.getShape()[1], mu.getShape()[0]}, 0, 1);
        DoubleTensor normalVariates = gaussian.sample(random);
        DoubleTensor x = choleskyDecomp.matrixMultiply(normalVariates);
        return x;
    }

}
