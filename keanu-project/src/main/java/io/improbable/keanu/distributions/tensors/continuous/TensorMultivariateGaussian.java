package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorMultivariateGaussian {

    private TensorMultivariateGaussian() {
    }

    public static DoubleTensor sample(DoubleTensor mu, DoubleTensor covariance, GaussianVertex gaussianVariates, KeanuRandom random) {
        DoubleTensor choleskyDecompOfCovariance = covariance.choleskyDecomposition().inverse();
        DoubleTensor samplesFromGaussianVariates = gaussianVariates.sample(random);
        return choleskyDecompOfCovariance.matrixMultiply(samplesFromGaussianVariates).plus(mu);
    }

    public static double logPdf(DoubleTensor mu, DoubleTensor covariance, DoubleTensor x) {
        double logOfCovarianceDeterminant = Math.log(covariance.determinant());
        DoubleTensor xMinusMu = x.minus(mu);
        DoubleTensor xMinusMuTransposed = xMinusMu.transpose();
        double kLog2Pi = mu.getShape()[0] * Math.log(2 * Math.PI);
        DoubleTensor covarianceInverted = covariance.inverse();
        DoubleTensor covInvTimesXMinusMu = covarianceInverted.matrixMultiply(xMinusMu);
        DoubleTensor foo = xMinusMuTransposed.matrixMultiply(covInvTimesXMinusMu);
        double scalar = foo.scalar();
        return (scalar + kLog2Pi + logOfCovarianceDeterminant) * (-0.5);
    }

}
