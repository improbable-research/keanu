package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorMultivariateGaussian {

    private TensorMultivariateGaussian() {
    }

    public static DoubleTensor sample(DoubleTensor mu, DoubleTensor covariance, GaussianVertex gaussianVariates, KeanuRandom random) {
        DoubleTensor choleskyDecompOfCovariance = covariance.choleskyDecomposition();
        DoubleTensor samplesFromGaussianVariates = gaussianVariates.sample(random);
        return choleskyDecompOfCovariance.matrixMultiply(samplesFromGaussianVariates).plus(mu);
    }

    public static double logPdf(DoubleTensor mu, DoubleTensor covariance, DoubleTensor x) {
        double numDimensions = mu.getShape()[0];
        double kLog2Pi = numDimensions * Math.log(2 * Math.PI);
        double logCovarianceDeterminant = Math.log(covariance.determinant());
        DoubleTensor xMinusMu = x.minus(mu);
        DoubleTensor xMinusMuTransposed = xMinusMu.transpose();
        DoubleTensor covarianceInverted = covariance.inverse();
        DoubleTensor covInvTimesXMinusMu = covarianceInverted.matrixMultiply(xMinusMu);
        double xMinusMuTransposedTimescovInvTimesXMinusMu = xMinusMuTransposed.matrixMultiply(covInvTimesXMinusMu).scalar();
        return -0.5 * (xMinusMuTransposedTimescovInvTimesXMinusMu + kLog2Pi + logCovarianceDeterminant);
    }

}
