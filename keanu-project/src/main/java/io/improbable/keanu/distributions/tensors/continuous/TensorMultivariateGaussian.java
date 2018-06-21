package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class TensorMultivariateGaussian {

    private TensorMultivariateGaussian() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor covariance, KeanuRandom random) {

    }

    public static DoubleTensor choleskyDecomposition(DoubleTensor covariance) {
        int length = covariance.getShape()[0];
        DoubleTensor cholesky = new Nd4jDoubleTensor(covariance.getShape());

        for (int i = 0; i < length; i++) {
            for (int k = 0; k < (i + 1); k++) {
                double sum = 0;
                for (int j = 0; j < k; j++) {
                    sum += covariance.getValue(i, j) * covariance.getValue(k, j);
                }
                double value = (i == k) ? Math.sqrt(covariance.getValue(i, i) - sum) :
                (1.0 / cholesky.getValue(k, k) * (covariance.getValue(i, k) - sum));
                cholesky.setValue(value, i, k);
            }
        }
        return cholesky;
    }

}
