package io.improbable.keanu.distributions.discrete;

import com.google.common.primitives.Ints;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;

import java.util.List;


public class Multinomial implements DiscreteDistribution {

    private final IntegerTensor n;
    private final DoubleTensor p;
    private final int numCategories;

    public static Multinomial withParameters(IntegerTensor n, DoubleTensor p) {
        return new Multinomial(n, p);
    }

    /**
     * @param n The number of draws from the variable
     * @param p The probability of observing each of the k values (which sum to 1)
     *          p is a Tensor whose last dimension must be of size k
     * @see <a href="https://en.wikipedia.org/wiki/Multinomial_distribution">Multinomial Distribution</a>
     * Generalisation of the Binomial distribution to variables with more than 2 possible values
     */
    private Multinomial(IntegerTensor n, DoubleTensor p) {
        numCategories = Ints.checkedCast(p.getShape()[p.getRank() - 1]);
        this.n = n;
        this.p = p;
    }


    @Override
    public IntegerTensor sample(long[] shape, KeanuRandom random) {

        long[] pBatchShape = TensorShape.selectDimensions(0, p.getRank() - 1, p.getShape());

        IntegerTensor broadcastedN = n.plus(IntegerTensor.zeros(pBatchShape));
        long[] broadcastResultShape = TensorShape.getBroadcastResultShape(TensorShape.concat(broadcastedN.getShape(), new long[]{1}), p.getShape());

        DoubleTensor pBroadcasted = p.plus(DoubleTensor.zeros(broadcastResultShape));

        Tensor.FlattenedView<Integer> nFlattened = broadcastedN.getFlattenedView();
        DoubleTensor reshapedP = pBroadcasted.reshape(-1, numCategories);

        List<DoubleTensor> sliced = reshapedP.sliceAlongDimension(0, 0, reshapedP.getShape()[0]);

        int length = TensorShape.getLengthAsInt(broadcastedN.getShape());
        int[] samples = new int[numCategories * length];

        for (int i = 0; i < length; i++) {
            double[] categoryProbabilities = sliced.get(i).asFlatDoubleArray();
            int[] sample = drawNTimes(nFlattened.getOrScalar(i), random, categoryProbabilities);
            System.arraycopy(sample, 0, samples, numCategories * i, sample.length);
        }

        //TODO make sure to use passed in shape
        return IntegerTensor.create(samples, shape);
    }

    private static int[] drawNTimes(int n, KeanuRandom random, double... categoryProbabilities) {
        int[] categoryDrawCounts = new int[categoryProbabilities.length];
        for (int i = 0; i < n; i++) {
            int index = draw(random, categoryProbabilities);
            categoryDrawCounts[index] += 1;
        }
        return categoryDrawCounts;
    }

    private static int draw(KeanuRandom random, double... categoryProbabilities) {
        double value = random.nextDouble();
        int index = 0;
        Double pCumulative = 0.;
        while (index < categoryProbabilities.length) {
            Double currentP = categoryProbabilities[index++];
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
        DoubleTensor gammaN = n.plus(1).toDouble().logGammaInPlace();
        DoubleTensor xLogP = p.log().timesInPlace(x.toDouble()).sum(-1);
        DoubleTensor gammaXs = x.plus(1).toDouble().logGammaInPlace().sum(-1);
        return xLogP.plusInPlace(gammaN).minusInPlace(gammaXs);
    }
}