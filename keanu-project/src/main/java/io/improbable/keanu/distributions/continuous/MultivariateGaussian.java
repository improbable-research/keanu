package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see <a href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution">Wikipedia</a>
 */
public class MultivariateGaussian implements ContinuousDistribution {

    private final DoubleTensor mean;
    private final DoubleTensor covariance;

    /**
     * @param mean       the mean of Multivariate Gaussian Distribution
     * @param covariance the covariance of Multivariate Gaussian Distribution
     */
    public static ContinuousDistribution withParameters(DoubleTensor mean, DoubleTensor covariance) {
        return new MultivariateGaussian(mean, covariance);
    }
    private MultivariateGaussian(DoubleTensor mean, DoubleTensor covariance) {
        this.mean = mean;
        this.covariance = covariance;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar(shape, mean.getShape());
        final DoubleTensor choleskyCov = covariance.choleskyDecomposition();
        final DoubleTensor variateSamples = random.nextGaussian(mean.getShape());
        final DoubleTensor covTimesVariates = mean.isScalar() ?
            choleskyCov.times(variateSamples) : choleskyCov.matrixMultiply(variateSamples);
        return covTimesVariates.plus(mean);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final double dimensions = mean.getShape()[0];
        final double kLog2Pi = dimensions * Math.log(2 * Math.PI);
        final double logCovDet = Math.log(covariance.determinant());
        DoubleTensor xMinusMu = x.minus(mean);
        DoubleTensor xMinusMuT = xMinusMu.transpose();
        DoubleTensor covInv = covariance.inverse();

        double scalar = mean.isScalar() ?
            covInv.times(xMinusMu).times(xMinusMuT).scalar() :
            xMinusMuT.matrixMultiply(covInv.matrixMultiply(xMinusMu)).scalar();

        return DoubleTensor.scalar(-0.5 * (scalar + kLog2Pi + logCovDet));
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }

}