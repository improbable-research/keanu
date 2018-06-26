package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorMultivariateGaussian {

    private TensorMultivariateGaussian() {
    }

    public static DoubleTensor sample(DoubleTensor mu, DoubleTensor covariance, KeanuRandom random) {
        DoubleTensor choleskyDecompOfCovariance = covariance.choleskyDecomposition();
        GaussianVertex gaussianVariates = new GaussianVertex(mu.getShape(), 0, 1);
        DoubleTensor samplesFromGaussianVariates = gaussianVariates.sample(random);
        return mu.plus(choleskyDecompOfCovariance.matrixMultiply(samplesFromGaussianVariates));
    }

    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor covariance, DoubleTensor x) {
        double logOfCovarianceDeterminant = Math.log(covariance.determinant());
        DoubleTensor xMinusMu = x.minus(mu);
        DoubleTensor xMinusMuTransposed = xMinusMu.transpose();
        double kLog2Pi = mu.getShape()[0] * Math.log(2 * Math.PI);
        DoubleTensor foo = covariance.reciprocal().times(xMinusMu);
        return (xMinusMuTransposed.times(foo).plusInPlace(kLog2Pi).plus(logOfCovarianceDeterminant)).times(-0.5);
    }

}
