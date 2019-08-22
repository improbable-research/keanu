package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.apache.commons.lang3.ArrayUtils;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShapeValidation.isBroadcastable;

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
    public DoubleTensor sample(long[] xShape, KeanuRandom random) {
        validateShapes(xShape, mu.getShape(), covariance.getShape());

        final long N = xShape[xShape.length - 1];
        final long[] xBatchShape = getBatch(xShape, 1);

        final DoubleTensor choleskyCov = covariance.choleskyDecomposition();
        final DoubleTensor variateSamples = random.nextGaussian(TensorShape.concat(xBatchShape, new long[]{N, 1}));
        final DoubleTensor covTimesVariates = choleskyCov.matrixMultiply(variateSamples);
        return covTimesVariates.reshape(xShape).plus(mu);
    }

    private void validateShapes(long[] xShape, long[] muShape, long[] covarianceShape) {
        if (xShape.length == 0) {
            throw new IllegalArgumentException(
                "X shape cannot be scalar. It must at least be a vector of length 1. Use a Gaussian distribution for scalar x."
            );
        }

        if (muShape.length == 0) {
            throw new IllegalArgumentException(
                "Mu shape cannot be scalar. It must at least be a vector of length 1."
            );
        }

        if (covarianceShape.length < 2) {
            throw new IllegalArgumentException(
                "Covariance matrix shape must be a matrix or a batch of matrices."
            );
        }

        final long xN = xShape[xShape.length - 1];
        final long muN = muShape[muShape.length - 1];
        final long covN = covarianceShape[covarianceShape.length - 1];
        final long covM = covarianceShape[covarianceShape.length - 2];

        if (xN != muN) {
            throw new IllegalArgumentException("Illegal x shape " + Arrays.toString(xShape) + " for mu shape " + Arrays.toString(muShape));
        }

        if (covN != xN) {
            throw new IllegalArgumentException("Illegal covariance shape " + Arrays.toString(covarianceShape) + " for x shape " + Arrays.toString(xShape));
        }

        if (covN != covM) {
            throw new IllegalArgumentException("Illegal covariance shape " + Arrays.toString(covarianceShape) + ". Shape must be square.");
        }

        if (xShape.length > 1 || muShape.length > 1 || covarianceShape.length > 2) {
            final long[] xBatchShape = getBatch(xShape, 1);
            final long[] muBatchShape = getBatch(muShape, 1);
            final long[] covBatchShape = getBatch(covarianceShape, 2);

            if (!isBroadcastable(xBatchShape, muBatchShape, covBatchShape)) {
                throw new IllegalArgumentException(
                    "x batch shape " + Arrays.toString(xBatchShape) +
                        " is not broadcastable with mu batch shape " + Arrays.toString(muBatchShape) +
                        " and covariance shape " + Arrays.toString(covBatchShape)
                );
            }
        }
    }

    private long[] getBatch(long[] shape, int eventRank) {
        return ArrayUtils.subarray(shape, 0, shape.length - eventRank);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        validateShapes(x.getShape(), mu.getShape(), covariance.getShape());

        try {
            final long[] xShape = x.getShape();
            final long N = xShape[xShape.length - 1];

            final DoubleTensor choleskyOfCovariance = covariance.choleskyDecomposition();
            final double kLog2Pi = N * LOG_2_PI;
            final DoubleTensor logCovDet = choleskyOfCovariance.diagPart().log().sum(-1).times(2);
            final DoubleTensor xMinusMu = x.minus(mu);

            DoubleTensor covInv = choleskyOfCovariance.choleskyInverse();
            covInv = covInv.plus(covInv.triLower(1).transpose());

            long[] xMinusMuShape = xMinusMu.getShape();
            long[] fromLeftShape = ArrayUtils.insert(xMinusMuShape.length - 1, xMinusMuShape, 1);
            long[] fromRightShape = ArrayUtils.add(xMinusMuShape, 1);

            DoubleTensor xmuCovXmu = xMinusMu.reshape(fromLeftShape)
                .matrixMultiply(
                    covInv.matrixMultiply(xMinusMu.reshape(fromRightShape))
                );

            xmuCovXmu = xmuCovXmu.reshape(getBatch(xmuCovXmu.getShape(), 2));

            return xmuCovXmu.plus(logCovDet.plus(kLog2Pi)).sum().times(-0.5);

        } catch (IllegalStateException ise) {
            return DoubleTensor.scalar(Double.NEGATIVE_INFINITY);
        }
    }

    public DoubleTensor[] dLogProb(DoubleTensor x, boolean wrtX, boolean wrtMu, boolean wrtCovariance) {
        DoubleTensor[] diff = new DoubleTensor[3];

        if (!wrtX && !wrtMu && !wrtCovariance) {
            return diff;
        }

        DoubleTensor covInv;
        try {
            covInv = covariance.choleskyDecomposition().choleskyInverse();
        } catch (IllegalStateException ise) {
            return diff;
        }
        covInv = covInv.plus(covInv.triLower(1).transpose());

        final DoubleTensor xMinusMu = x.minus(mu);

        final long[] xBatchShape = getBatch(x.getShape(), 1);
        final long[] muBatchShape = getBatch(mu.getShape(), 1);
        final long[] covBatchShape = getBatch(covariance.getShape(), 2);
        final long[] resultBatchShape = TensorShape.getBroadcastResultShape(xBatchShape, muBatchShape, covBatchShape);

        final long dims = x.getShape()[x.getShape().length - 1];

        if (wrtMu || wrtX) {

            final DoubleTensor dLogProbWrtMuForBatch = covInv.matrixMultiply(
                xMinusMu.reshape(TensorShape.concat(xBatchShape, new long[]{dims, 1}))
            ).reshape(TensorShape.concat(resultBatchShape, new long[]{dims}));

            if (wrtX) {
                diff[0] = sumOverBatch(dLogProbWrtMuForBatch, x.getShape()).unaryMinus();
            }

            if (wrtMu) {
                diff[1] = sumOverBatch(dLogProbWrtMuForBatch, mu.getShape());
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

            diff[2] = sumOverBatch(dLogProbWrtCovarianceForBatch, covariance.getShape());
        }

        return diff;

    }

    private static DoubleTensor sumOverBatch(DoubleTensor batched, long[] targetShape) {
        return batched.sum(TensorShape.dimensionRange(0, batched.getShape().length - targetShape.length));
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
