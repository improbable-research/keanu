package io.improbable.keanu.distributions.discrete;

import com.google.common.base.Preconditions;
import com.google.common.primitives.Ints;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

import java.util.Arrays;

import static io.improbable.keanu.tensor.TensorShapeValidation.isBroadcastable;


public class Multinomial implements DiscreteDistribution {

    private final IntegerTensor n;
    private final DoubleTensor p;
    private final int numCategories;
    private boolean validationEnabled;

    public static Multinomial withParameters(IntegerTensor n, DoubleTensor p) {
        return new Multinomial(n, p, false);
    }

    public static Multinomial withParameters(IntegerTensor n, DoubleTensor p, boolean validationEnabled) {
        return new Multinomial(n, p, validationEnabled);
    }

    /**
     * @param n The number of draws from the variable
     * @param p The probability of observing each of the k values (which sum to 1)
     *          p is a Tensor whose last dimension must be of size k
     * @see <a href="https://en.wikipedia.org/wiki/Multinomial_distribution">Multinomial Distribution</a>
     * Generalisation of the Binomial distribution to variables with more than 2 possible values
     */
    private Multinomial(IntegerTensor n, DoubleTensor p, boolean validationEnabled) {
        numCategories = Ints.checkedCast(p.getShape()[p.getRank() - 1]);
        this.n = n;
        this.p = p;
        this.validationEnabled = validationEnabled;
    }

    @Override
    public IntegerTensor sample(long[] shape, KeanuRandom random) {

        if (validationEnabled) {
            validateBroadcastShapes(shape, n.getShape(), p.getShape());
            validateProbabilities(p);
            validateN(n);
        }

        long[] sampleBatchShape = TensorShape.selectDimensions(0, shape.length - 1, shape);

        IntegerTensor broadcastedN = n.plus(IntegerTensor.zeros(sampleBatchShape));
        long[] broadcastResultShape = TensorShape.getBroadcastResultShape(
            TensorShape.concat(broadcastedN.getShape(), new long[]{1}), p.getShape()
        );

        DoubleTensor broadcastedP = p.plus(DoubleTensor.zeros(broadcastResultShape));
        DoubleTensor reshapedP = broadcastedP.reshape(-1, numCategories);

        double[] flatP = reshapedP.asFlatDoubleArray();
        int[] flatN = broadcastedN.asFlatIntegerArray();

        int sampleCount = TensorShape.getLengthAsInt(broadcastedN.getShape());
        int[] samples = new int[numCategories * sampleCount];

        for (int i = 0; i < sampleCount; i++) {

            int pIndex = i * numCategories;
            drawNTimes(flatN[i], random, samples, pIndex, flatP, pIndex, numCategories);
        }

        return IntegerTensor.create(samples, shape);
    }

    private static void drawNTimes(int n, KeanuRandom random, int[] sample, int sampleIndex, double[] p, int pIndex, int pCount) {
        for (int i = 0; i < n; i++) {
            int index = draw(random, p, pIndex, pCount);
            sample[sampleIndex + index] += 1;
        }
    }

    private static int draw(KeanuRandom random, double[] p, int pIndex, int pCount) {
        double value = random.nextDouble();
        int index = 0;
        Double pCumulative = 0.;
        while (index < pCount) {
            Double currentP = p[pIndex + index++];
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
            validateProbabilities(p);
            validateN(n);
            validateX(x, n, p);
        }

        DoubleTensor gammaN = n.plus(1).toDouble().logGammaInPlace();
        DoubleTensor xLogP = p.log().timesInPlace(x.toDouble()).sum(-1);
        DoubleTensor gammaXs = x.plus(1).toDouble().logGammaInPlace().sum(-1);
        return xLogP.plusInPlace(gammaN).minusInPlace(gammaXs);
    }

    private static void validateProbabilities(DoubleTensor p) {
        if (p.isScalar()) {
            throw new IllegalArgumentException("Probabilities must be a vector or a tensor with rank >= 1");
        }

        final boolean pRangeValidated = p.greaterThan(0.0).allTrue() && p.lessThan(1.0).allTrue();
        if (!pRangeValidated) {
            throw new IllegalArgumentException(
                "Probabilities must be > 0 < 1 but were " + Arrays.toString(p.asFlatDoubleArray())
            );
        }

        final DoubleTensor pSum = p.sum(-1);
        final boolean pSumValidated = pSum.equalsWithinEpsilon(
            DoubleTensor.create(1.0, pSum.getShape()), 1e-8
        );
        if (!pSumValidated) {
            throw new IllegalArgumentException(
                "Probabilities must sum to 1 but summed to " + Arrays.toString(pSum.asFlatDoubleArray())
            );
        }
    }

    private static void validateN(IntegerTensor n) {
        final boolean nRangeValidated = n.greaterThanOrEqual(0).allTrue();
        if (!nRangeValidated) {
            throw new IllegalArgumentException("Number of trials (n) must be non-negative.");
        }
    }

    private static void validateX(IntegerTensor x, IntegerTensor n, DoubleTensor p) {
        final boolean xRangeValidated = x.greaterThanOrEqual(0).allTrue() && x.lessThanOrEqual(n).allTrue();
        if (!xRangeValidated) {
            throw new IllegalArgumentException("x must be >= 0 and <= n");
        }

        long kAccordingToP = p.getShape()[p.getRank() - 1];
        long kAccordingToX = x.isScalar() ? 0 : x.getShape()[x.getRank() - 1];
        Preconditions.checkArgument(
            kAccordingToX == kAccordingToP,
            "x shape must have far right dimension matching number of categories k " + kAccordingToP +
                " but had " + kAccordingToX + " categories."
        );

        final IntegerTensor xSum = x.sum(-1);
        final boolean xSumValidated = xSum.elementwiseEquals(n).allTrue();
        if (!xSumValidated) {
            throw new IllegalArgumentException(
                "The sum of x " + Arrays.toString(xSum.asFlatArray()) +
                    " must equal n " + Arrays.toString(n.asFlatDoubleArray())
            );
        }
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
            );
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