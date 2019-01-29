package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LogProbGraph.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class MultivariateGaussian implements ContinuousDistribution {

    private static final double LOG_2_PI = Math.log(2 * Math.PI);
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
        final double kLog2Pi = dimensions * LOG_2_PI;
        final double logCovDet = Math.log(covariance.determinant());
        DoubleTensor xMinusMu = x.minus(mu);
        DoubleTensor xMinusMuT = xMinusMu.transpose();
        DoubleTensor covInv = covariance.matrixInverse();

        double scalar = isUnivariate() ?
            covInv.times(xMinusMu).times(xMinusMuT).scalar() :
            xMinusMuT.matrixMultiply(covInv.matrixMultiply(xMinusMu)).scalar();

        return DoubleTensor.scalar(-0.5 * (scalar + kLog2Pi + logCovDet));
    }

    public static DoubleVertex logProbGraph(DoublePlaceholderVertex x, DoublePlaceholderVertex mu, DoublePlaceholderVertex covariance) {
        final double dimensions = numberOfDimensions(mu);
        final double kLog2Pi = dimensions * LOG_2_PI;
        final DoubleVertex logCovDet = covariance.matrixDeterminant().log();
        DoubleVertex xMinusMu = x.minus(mu);
        DoubleVertex xMinusMuT = xMinusMu.permute(1, 0);
        DoubleVertex covInv = covariance.matrixInverse();

        boolean isUnivariate = dimensions == 1;
        DoubleVertex scalar = isUnivariate ?
            covInv.times(xMinusMu).times(xMinusMuT):
            xMinusMuT.matrixMultiply(covInv.matrixMultiply(xMinusMu));

        return scalar.plus(kLog2Pi).plus(logCovDet).times(-0.5).slice(0, 0);
    }

    private boolean isUnivariate() {
        return numberOfDimensions() == 1;
    }

    private static long numberOfDimensions(DoublePlaceholderVertex mu) {
        return mu.getShape()[0];
    }

    private long numberOfDimensions() {
        return mu.getShape()[0];
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }
}
