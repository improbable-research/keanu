package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import org.apache.commons.lang3.ArrayUtils;

import java.util.List;
import java.util.stream.Collectors;


public class Multinomial implements DiscreteDistribution {

    private final IntegerTensor n;
    private final DoubleTensor p;
    private final long numCategories;

    public static Multinomial withParameters(IntegerTensor n, DoubleTensor p) {
        return new Multinomial(n, p);
    }

    /**
     * @param n The number of draws from the variable
     * @param p The probability of observing each of the k values (which sum to 1)
     *          p is a Tensor whose first dimension must be of size k
     * @see <a href="https://en.wikipedia.org/wiki/Multinomial_distribution">Multinomial Distribution</a>
     * Generalisation of the Binomial distribution to variables with more than 2 possible values
     */
    private Multinomial(IntegerTensor n, DoubleTensor p) {
        numCategories = p.getShape()[0];
        this.n = n;
        this.p = p;
    }


    @Override
    public IntegerTensor sample(long[] shape, KeanuRandom random) {

        Tensor.FlattenedView<Integer> nFlattened = n.getFlattenedView();
        List<DoubleTensor> sliced = p.sliceAlongDimension(0, 0, numCategories);

        int length = TensorShape.getLengthAsInt(shape);
        int[] samples = new int[0];

        for (int i = 0; i < length; i++) {
            final int j = i;
            List<Double> categoryProbabilities = sliced.stream().map(p -> p.getFlattenedView().getOrScalar(j)).collect(Collectors.toList());
            int[] sample = drawNTimes(nFlattened.getOrScalar(i), random, categoryProbabilities.toArray(new Double[0]));
            samples = ArrayUtils.addAll(samples, sample);
        }
        return constructSampleTensor(shape, samples);
    }

    /**
     * This method is necessary because I've constructed a flat array by concatenation samples of size k
     * So for example, if the shape of n is [a, b]
     * then I've now got a tensor of shape [a, b, k]
     * which I need to convert to a tensor of shape [k, a, b]
     * by doing a slice in the highest dimension and then concatenating again
     *
     * @param shape   the desired shape, not including the probabilities dimension
     * @param samples the flat array of samples
     * @return
     */
    private IntegerTensor constructSampleTensor(long[] shape, int[] samples) {
        long[] outputShape = shape;
        if (shape[0] == 1) {
            outputShape = ArrayUtils.remove(outputShape, 0);
        }
        IntegerTensor abkTensor = IntegerTensor.create(samples, ArrayUtils.add(outputShape, numCategories));
        int[] kabArray = new int[]{};
        for (int category = 0; category < numCategories; category++) {
            IntegerTensor abTensor = abkTensor.slice(outputShape.length, category);
            kabArray = ArrayUtils.addAll(kabArray, abTensor.asFlatIntegerArray());
        }

        return IntegerTensor.create(kabArray, ArrayUtils.insert(0, outputShape, numCategories));
    }

    private static int[] drawNTimes(int n, KeanuRandom random, Double... categoryProbabilities) {
        int[] categoryDrawCounts = new int[categoryProbabilities.length];
        for (int i = 0; i < n; i++) {
            int index = draw(random, categoryProbabilities);
            categoryDrawCounts[index] += 1;
        }
        return categoryDrawCounts;
    }

    private static int draw(KeanuRandom random, Double... categoryProbabilities) {
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
    public DoubleTensor logProb(IntegerTensor k) {
        DoubleTensor gammaN = n.plus(1).toDouble().logGammaInPlace();
        DoubleTensor gammaKs = k.plus(1).toDouble().logGammaInPlace().sum(0);
        DoubleTensor kLogP = p.log().timesInPlace(k.toDouble()).sum(0);
        return kLogP.plusInPlace(gammaN).minusInPlace(gammaKs);
    }
}