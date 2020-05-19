package io.improbable.keanu.distributions.discrete;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShapeValidation.isBroadcastable;


public class Multinomial implements DiscreteDistribution {

    private final IntegerTensor n;
    private final DoubleTensor p;
    private final int k;
    private boolean validationEnabled;

    public static Multinomial withParameters(IntegerTensor n, DoubleTensor p) {
        return new Multinomial(n, p, false);
    }

    public static Multinomial withParameters(IntegerTensor n, DoubleTensor p, boolean validationEnabled) {
        return new Multinomial(n, p, validationEnabled);
    }

    /**
     * @param n The number of draws from the variable
     * @param p The probability of observing each of the k values (which will be normalized to sum to 1)
     *          p is a Tensor whose last dimension must be of size k
     * @see <a href="https://en.wikipedia.org/wiki/Multinomial_distribution">Multinomial Distribution</a>
     * Generalisation of the Binomial distribution to variables with more than 2 possible values
     */
    private Multinomial(IntegerTensor n, DoubleTensor p, boolean validationEnabled) {
        k = Ints.checkedCast(p.getShape()[p.getRank() - 1]);
        this.n = n;
        this.p = p;
        this.validationEnabled = validationEnabled;

        if (validationEnabled) {
            validateProbabilities(p);
            validateN(n);
        }
    }

    @Override
    public IntegerTensor sample(long[] shape, KeanuRandom random) {

        if (validationEnabled) {
            validateBroadcastShapes(shape, n.getShape(), p.getShape());
            validateXShape(shape, p.getShape());
        }

        final long[] sampleBatchShape = TensorShape.selectDimensions(0, shape.length - 1, shape);

        final IntegerTensor broadcastedN = n.broadcast(sampleBatchShape);
        final long[] broadcastResultShape = TensorShape.getBroadcastResultShape(
            TensorShape.concat(broadcastedN.getShape(), new long[]{1}), p.getShape()
        );

        final DoubleTensor pNormalized = p.div(p.sum(-1).expandDims(-1));
        final DoubleTensor broadcastedP = pNormalized.broadcast(broadcastResultShape);

        final double[] flatP = broadcastedP.asFlatDoubleArray();
        final int[] flatN = broadcastedN.asFlatIntegerArray();

        final int sampleCount = flatN.length;
        final int[] samples = new int[k * sampleCount];

        for (int i = 0; i < sampleCount; i++) {
            final int positionByK = i * k;
            drawNTimes(flatN[i], random, samples, positionByK, flatP, positionByK, k);
        }

        return IntegerTensor.create(samples, shape);
    }

    private static void drawNTimes(int n, KeanuRandom random, int[] sample, int sampleIndex, double[] p, int pIndex, int pCount) {
        for (int i = 0; i < n; i++) {
            int drawnIndex = draw(random, p, pIndex, pCount);
            sample[sampleIndex + drawnIndex] += 1;
        }
    }

    private static int draw(KeanuRandom random, double[] p, int pIndex, int pCount) {
        double value = random.nextDouble();
        int index = 0;
        double pCumulative = 0.;
        while (index < pCount) {
            double currentP = p[pIndex + index];
            index++;
            if (currentP == 0.) {
                continue;
            }
            pCumulative += currentP;
            if (pCumulative >= value) {
                break;
            }
        }
        return index - 1;
    }

    @Override
    public DoubleTensor logProb(IntegerTensor x) {
        if (validationEnabled) {
            validateBroadcastShapes(x.getShape(), n.getShape(), p.getShape());
            validateX(x, n, p);
        }

        final DoubleTensor pNormalized = p.div(p.sum(-1).expandDims(-1));
        final DoubleTensor gammaN = n.plus(1).toDouble().logGammaInPlace();
        final DoubleTensor xLogP = pNormalized.safeLogTimesInPlace(x.toDouble()).sum(-1);
        final DoubleTensor gammaXs = x.plus(1).toDouble().logGammaInPlace().sum(-1);
        return xLogP.plusInPlace(gammaN).minusInPlace(gammaXs);
    }

    public DoubleTensor dLogProb(IntegerTensor x) {
        if (validationEnabled) {
            validateBroadcastShapes(x.getShape(), n.getShape(), p.getShape());
            validateX(x, n, p);
        }

        final DoubleTensor pNormalized = p.div(p.sum(-1).expandDims(-1));
        final DoubleTensor dLogProbWrtP = x.toDouble().divInPlace(pNormalized);

        return dLogProbWrtP;
    }

    private static void validateProbabilities(DoubleTensor p) {
        if (p.isScalar()) {
            throw new IllegalArgumentException("Probabilities must be a vector or a tensor with rank >= 1");
        }

        final boolean pRangeValidated = p.greaterThanOrEqual(0.0).allTrue().scalar();
        if (!pRangeValidated) {
            throw new IllegalArgumentException(
                "Probabilities must be >= 0 but were " + Arrays.toString(p.asFlatDoubleArray())
            );
        }
    }

    private static void validateN(IntegerTensor n) {
        final boolean nRangeValidated = n.greaterThanOrEqual(0).allTrue().scalar();
        if (!nRangeValidated) {
            throw new IllegalArgumentException("Number of trials (n) must be non-negative.");
        }
    }

    private static void validateX(IntegerTensor x, IntegerTensor n, DoubleTensor p) {
        final boolean xRangeValidated = x.greaterThanOrEqual(0).allTrue().scalar() &&
            x.lessThanOrEqual(n.reshape(Longs.concat(n.getShape(), new long[]{1}))).allTrue().scalar();

        if (!xRangeValidated) {
            throw new IllegalArgumentException("x must be >= 0 and <= n");
        }

        validateXShape(x.getShape(), p.getShape());

        final IntegerTensor xSum = x.sum(-1);
        final boolean xSumValidated = xSum.elementwiseEquals(n).allTrue().scalar();
        if (!xSumValidated) {
            throw new IllegalArgumentException(
                "The sum of x " + Arrays.toString(xSum.asFlatArray()) +
                    " must equal n " + Arrays.toString(n.asFlatDoubleArray())
            );
        }
    }

    private static void validateXShape(long[] xShape, long[] pShape) {

        long kAccordingToP = pShape[pShape.length - 1];
        long kAccordingToX = xShape.length == 0 ? 0 : xShape[xShape.length - 1];
        Preconditions.checkArgument(
            kAccordingToX == kAccordingToP,
            "x shape must have far right dimension matching number of categories k " + kAccordingToP +
                " but had " + kAccordingToX + " categories."
        );
    }

    private static void validateBroadcastShapes(long[] xShape, long[] nShape, long[] pShape) {

        long[] broadcastResultShape;
        try {
            broadcastResultShape = TensorShape.getBroadcastResultShape(
                pShape, TensorShape.concat(nShape, new long[]{1})
            );
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                "p shape " + Arrays.toString(pShape) + " incompatible with n shape " +
                    Arrays.toString(nShape)
                , e);
        }

        if (!isBroadcastable(broadcastResultShape, xShape)) {
            throw new IllegalArgumentException("Shape " +
                Arrays.toString(xShape) + " is incompatible with n shape " +
                Arrays.toString(nShape) + " and p shape " + Arrays.toString(pShape) +
                ". It must be broadcastable with " + Arrays.toString(broadcastResultShape)
            );
        }

    }
}