package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

public class MultivariateGaussian implements ContinuousDistribution {

    private final DoubleTensor mu;
    private final DoubleTensor covariance;

    public static ContinuousDistribution withParameters(DoubleTensor mu, DoubleTensor covariance) {
        return new MultivariateGaussian(mu, covariance);
    }

    private MultivariateGaussian(DoubleTensor mu, DoubleTensor covariance) {
        this.mu = mu;
        this.covariance = covariance;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        TensorShapeValidation.checkTensorsMatchNonLengthOneShapeOrAreLengthOne(shape, mu.getShape());
        final DoubleTensor choleskyCov = covariance.choleskyDecomposition();
        final DoubleTensor variateSamples = random.nextGaussian(mu.getShape());
        final DoubleTensor covTimesVariates = isUnivariate() ?
            choleskyCov.times(variateSamples) : choleskyCov.matrixMultiply(variateSamples);
        return covTimesVariates.plus(mu);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final double dimensions = numberOfDimensions();
        final double kLog2Pi = dimensions * Math.log(2 * Math.PI);
        final double logCovDet = Math.log(covariance.determinant());
        DoubleTensor xMinusMu = x.minus(mu);
        DoubleTensor xMinusMuT = xMinusMu.transpose();
        DoubleTensor covInv = covariance.matrixInverse();

        double scalar = isUnivariate() ?
            covInv.times(xMinusMu).times(xMinusMuT).scalar() :
            xMinusMuT.matrixMultiply(covInv.matrixMultiply(xMinusMu)).scalar();

        return DoubleTensor.scalar(-0.5 * (scalar + kLog2Pi + logCovDet));
    }

    private boolean isUnivariate() {
        return numberOfDimensions() == 1;
    }

    private long numberOfDimensions() {
        return mu.getShape()[0];
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }
}
