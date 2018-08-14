package io.improbable.keanu.distributions.discrete;

import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.special.Gamma;
import org.nd4j.linalg.util.ArrayUtil;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;


public class Multinomial implements DiscreteDistribution {

    private final IntegerTensor n;
    private final DoubleTensor p;
    private final int numCategories;

    public static Multinomial withParameters(IntegerTensor n, DoubleTensor p) {
        return new Multinomial(n, p);
    }

    /**
     * https://en.wikipedia.org/wiki/Multinomial_distribution
     * Generalisation of the Binomial distribution to variables with >2 possible values
     *
     * @param n The number of draws from the variable
     * @param p The probability of observing each of the k values (which sum to 1)
     *          p is a Tensor whose first dimension must be of size k
     */
    private Multinomial(IntegerTensor n, DoubleTensor p) {
        Preconditions.checkArgument(
            p.sum(0).elementwiseEquals(DoubleTensor.ones(n.getShape())).allTrue(),
            "Probabilities must sum to one"
        );

        numCategories = p.getShape()[0];
        for (int category = 0; category < numCategories; category++) {
            TensorShapeValidation.checkAllShapesMatch(n.getShape(), p.slice(0, category).getShape());
        }
        this.n = n;
        this.p = p;
    }

    @Override
    public IntegerTensor sample(int[] shape, KeanuRandom random) {
        p.sum(0);
        Tensor.FlattenedView<Integer> nFlattened = n.getFlattenedView();
        List<Tensor.FlattenedView<Double>> pFlattened = Lists.newArrayList();

        for (int category = 0; category < numCategories; category++) {
            DoubleTensor p_i = p.slice(0, category);
            pFlattened.add(p_i.getFlattenedView());
        }

        int length = ArrayUtil.prod(shape);
        int[] samples = new int[0];

        for (int i = 0; i < length; i++) {
            final int j = i;
            List<Double> ps = pFlattened.stream().map(p -> p.getOrScalar(j)).collect(Collectors.toList());
            int[] sample = sample(nFlattened.getOrScalar(i), random, ps.toArray(new Double[0]));
            samples = ArrayUtils.addAll(samples, sample);
        }
        int[] outputShape = shape;
        if (shape[0] == 1) {
            outputShape = ArrayUtils.remove(outputShape, 0);
        }
        outputShape = ArrayUtils.add(outputShape, numCategories);
        return IntegerTensor.create(samples, outputShape).transpose();

    }

    private static int[] sample(int n, KeanuRandom random, Double... p) {
        int[] counts = new int[p.length];
        for (int i = 0; i < n; i++) {
            int index = sample(random, p);
            counts[index] += 1;
        }
        return counts;
    }

    private static int sample(KeanuRandom random, Double... p) {
        double value = random.nextDouble();
        int index = 0;
        Double pCumulative = 0.;
        while (index < p.length) {
            Double currentP = p[index++];
            if (currentP == 0.) {
                continue;
            }
            pCumulative += currentP;
            if (pCumulative >= value) {
                break;
            }
        }
        return index-1;
    }

    @Override
    public DoubleTensor logProb(IntegerTensor k) {
        TensorShapeValidation.checkAllShapesMatch(
            String.format("k: %s, p: %s", k, p.sum(0)),
            k.getShape(), new int[] {p.getShape()[0], 1}
        );
        Preconditions.checkArgument(
            k.sum(0).elementwiseEquals(this.n).allTrue(),
            String.format("Inputs %s must sum to n = %s", k, this.n)
        );

        DoubleTensor gammaN = n.plus(1).toDouble().apply(Gamma::gamma).log();
        DoubleTensor gammaKs = k.plus(1).toDouble().apply(Gamma::gamma).log().sum(0);
        DoubleTensor kLogP = p.log().times(k.toDouble()).sum(0);
        return kLogP.plus(gammaN).minus(gammaKs);
    }
}
