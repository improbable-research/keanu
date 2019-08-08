package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

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
        validateXShape(shape, mu.getShape());
        long[] batchShape = ArrayUtils.subarray(shape, 0, shape.length - 1);
        final DoubleTensor choleskyCov = covariance.choleskyDecomposition();
        final DoubleTensor variateSamples = random.nextGaussian(TensorShape.concat(batchShape, new long[]{mu.getShape()[0], 1}));
        final DoubleTensor covTimesVariates = choleskyCov.matrixMultiply(variateSamples);
        return covTimesVariates.reshape(shape).plus(mu);
    }

    private void validateXShape(long[] xShape, long[] muShape) {
        if (xShape[xShape.length - 1] != muShape[muShape.length - 1]) {
            throw new IllegalArgumentException("Illegal x shape " + Arrays.toString(xShape) + " for mu shape " + Arrays.toString(muShape));
        }
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final double kLog2Pi = mu.getShape()[0] * LOG_2_PI;
        final double logCovDet = covariance.matrixDeterminant().log().scalar();
        final DoubleTensor xMinusMu = x.minus(mu);
        final DoubleTensor covInv = covariance.matrixInverse();

        long[] xMinusMuShape = xMinusMu.getShape();
        long[] fromLeftShape = ArrayUtils.insert(xMinusMuShape.length - 1, xMinusMuShape, 1);
        long[] fromRightShape = ArrayUtils.add(xMinusMuShape, 1);

        final DoubleTensor xmuCovXmu = xMinusMu.reshape(fromLeftShape)
            .matrixMultiply(
                covInv.matrixMultiply(xMinusMu.reshape(fromRightShape))
            );

        return xmuCovXmu.plus(kLog2Pi + logCovDet).sum().times(-0.5);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }

    public static DoubleVertex logProbGraph(DoublePlaceholderVertex x, DoublePlaceholderVertex mu, DoublePlaceholderVertex covariance) {
        final double kLog2Pi = mu.getShape()[0] * LOG_2_PI;
        final DoubleVertex logCovDet = covariance.matrixDeterminant().log();
        final DoubleVertex xMinusMu = x.minus(mu);
        final DoubleVertex covInv = covariance.matrixInverse();

        long[] xMinusMuShape = xMinusMu.getShape();
        long[] fromLeftShape = ArrayUtils.insert(xMinusMuShape.length - 1, xMinusMuShape, 1);
        long[] fromRightShape = ArrayUtils.add(xMinusMuShape, 1);

        final DoubleVertex xmuCovXmu = xMinusMu.reshape(fromLeftShape)
            .matrixMultiply(
                covInv.matrixMultiply(xMinusMu.reshape(fromRightShape))
            );

        return xmuCovXmu.plus(kLog2Pi).plus(logCovDet).sum().times(-0.5);
    }

}
