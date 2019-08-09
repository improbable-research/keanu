package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
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

    public static MultivariateGaussian withParameters(DoubleTensor mu, DoubleTensor covariance) {
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

    public DoubleTensor[] dLogProb(DoubleTensor x, boolean wrtX, boolean wrtMu, boolean wrtCovariance) {

        final DoubleTensor covInv = covariance.matrixInverse();
        final DoubleTensor xMinusMu = x.minus(mu);

        final long[] batchShape = ArrayUtils.subarray(x.getShape(), 0, x.getShape().length - 1);
        final long dims = x.getShape()[x.getShape().length - 1];

        DoubleTensor[] diff = new DoubleTensor[3];
        if (wrtMu || wrtX) {

            final DoubleTensor dLogProbWrtMuForBatch = covInv.matrixMultiply(
                xMinusMu.reshape(TensorShape.concat(batchShape, new long[]{dims, 1}))
            ).reshape(TensorShape.concat(batchShape, new long[]{dims}));

            if (wrtX) {
                final DoubleTensor dLogProbWrtX = dLogProbWrtMuForBatch
                    .sum(TensorShape.dimensionRange(
                        0,
                        dLogProbWrtMuForBatch.getShape().length - x.getShape().length
                    )).unaryMinus();

                diff[0] = dLogProbWrtX;
            }

            if (wrtMu) {
                final DoubleTensor dLogProbWrtMu = dLogProbWrtMuForBatch
                    .sum(TensorShape.dimensionRange(
                        0,
                        dLogProbWrtMuForBatch.getShape().length - mu.getShape().length
                    ));
                diff[1] = dLogProbWrtMu;
            }
        }

        if (wrtCovariance) {

            final DoubleTensor covInvTranspose = covInv.transpose();

            final DoubleTensor xMinusMuMatrix = xMinusMu.reshape(TensorShape.concat(xMinusMu.getShape(), new long[]{1}));

            final DoubleTensor dLogProbWrtCovarianceForBatch = covInvTranspose
                .matrixMultiply(
                    xMinusMuMatrix.matrixMultiply(xMinusMuMatrix.swapAxis(-2, -1)).matrixMultiply(covInvTranspose)
                )
                .minus(covInvTranspose)
                .times(0.5);

            final DoubleTensor dLogProbWrtCovariance = dLogProbWrtCovarianceForBatch
                .sum(TensorShape.dimensionRange(
                    0,
                    dLogProbWrtCovarianceForBatch.getShape().length - covariance.getShape().length
                ));

            diff[2] = dLogProbWrtCovariance;
        }

        return diff;
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
