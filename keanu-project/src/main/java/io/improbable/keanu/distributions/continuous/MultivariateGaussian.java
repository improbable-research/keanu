package io.improbable.keanu.distributions.continuous;

import com.google.common.base.Preconditions;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

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
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        for (int i = 0; i < shape.length; i++) {
            Preconditions.checkArgument(mu.getShape()[i] == shape[i],
                "shape must match mu's shape");
        }
        final DoubleTensor choleskyCov = covariance.choleskyDecomposition();
        final DoubleTensor variateSamples = random.nextGaussian(mu.getShape());
        final DoubleTensor covTimesVariates = mu.isScalar() ?
            choleskyCov.times(variateSamples) : choleskyCov.matrixMultiply(variateSamples);
        return covTimesVariates.plus(mu);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final double dimensions = mu.getShape()[0];
        final double kLog2Pi = dimensions * Math.log(2 * Math.PI);
        final double logCovDet = Math.log(covariance.determinant());
        DoubleTensor xMinusMu = x.minus(mu);
        DoubleTensor xMinusMuT = xMinusMu.transpose();
        DoubleTensor covInv = covariance.inverse();

        double scalar = mu.isScalar() ?
            covInv.times(xMinusMu).times(xMinusMuT).scalar() :
            xMinusMuT.matrixMultiply(covInv.matrixMultiply(xMinusMu)).scalar();

        return DoubleTensor.scalar(-0.5 * (scalar + kLog2Pi + logCovDet));
    }

    @Override
    public ParameterMap<DoubleTensor> dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }
}
