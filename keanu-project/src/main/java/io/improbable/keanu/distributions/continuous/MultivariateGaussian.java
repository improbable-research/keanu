package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class MultivariateGaussian {

    private MultivariateGaussian() {
    }

    public static DoubleTensor sample(DoubleTensor mu, DoubleTensor covariance, KeanuRandom random) {
        final DoubleTensor choleskyCov = covariance.choleskyDecomposition();
        final DoubleTensor variateSamples = random.nextGaussian(mu.getShape());
        final DoubleTensor covTimesVariates = mu.isScalar() ?
            choleskyCov.times(variateSamples) : choleskyCov.matrixMultiply(variateSamples);
        return covTimesVariates.plus(mu);
    }

    public static double logPdf(DoubleTensor mu, DoubleTensor covariance, DoubleTensor x) {
        final double dimensions = mu.getShape()[0];
        final double kLog2Pi = dimensions * Math.log(2 * Math.PI);
        final double logCovDet = Math.log(covariance.determinant());
        DoubleTensor xMinusMu = x.minus(mu);
        DoubleTensor xMinusMuT = xMinusMu.transpose();
        DoubleTensor covInv = covariance.inverse();

        double scalar = mu.isScalar() ?
            covInv.times(xMinusMu).times(xMinusMuT).scalar() :
            xMinusMuT.matrixMultiply(covInv.matrixMultiply(xMinusMu)).scalar();

        return -0.5 * (scalar + kLog2Pi + logCovDet);
    }
}
