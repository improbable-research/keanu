package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class MultivariateGaussian implements ContinuousDistribution {

    private final DoubleTensor location;
    private final DoubleTensor scale;

    /**
     * <h3>Multivariate Gaussian (Normal) Distribution</h3>
     *
     * @param location shifts the distribution; mean
     * @param scale    stretches/shrinks the distribution; covariance
     * @see <a href="https://en.wikipedia.org/wiki/Multivariate_normal_distribution">Wikipedia</a>
     */
    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor scale) {
        return new MultivariateGaussian(location, scale);
    }
    private MultivariateGaussian(DoubleTensor location, DoubleTensor scale) {
        this.location = location;
        this.scale = scale;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        TensorShapeValidation.checkTensorsMatchNonScalarShapeOrAreScalar(shape, location.getShape());
        final DoubleTensor choleskyCov = scale.choleskyDecomposition();
        final DoubleTensor variateSamples = random.nextGaussian(location.getShape());
        final DoubleTensor covTimesVariates = location.isScalar() ?
            choleskyCov.times(variateSamples) : choleskyCov.matrixMultiply(variateSamples);
        return covTimesVariates.plus(location);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final double dimensions = location.getShape()[0];
        final double kLog2Pi = dimensions * Math.log(2 * Math.PI);
        final double logCovDet = Math.log(scale.determinant());
        DoubleTensor xMinusMean = x.minus(location);
        DoubleTensor xMinusMeanT = xMinusMean.transpose();
        DoubleTensor covInv = scale.inverse();

        double scalar = location.isScalar() ?
            covInv.times(xMinusMean).times(xMinusMeanT).scalar() :
            xMinusMeanT.matrixMultiply(covInv.matrixMultiply(xMinusMean)).scalar();

        return DoubleTensor.scalar(-0.5 * (scalar + kLog2Pi + logCovDet));
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }

}