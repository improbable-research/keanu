package io.improbable.keanu.distributions.discrete;

import static org.apache.commons.math3.util.CombinatoricsUtils.factorial;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.2.8 page 49"
 */
public class Poisson implements DiscreteDistribution {

    private final DoubleTensor rate;

    /**
     * @param rate rate of occurrence
     */
    public static DiscreteDistribution withParameters(DoubleTensor rate) {
        return new Poisson(rate);
    }

    private Poisson(DoubleTensor rate) {
        this.rate = rate;
    }

    @Override
    public IntegerTensor sample(int[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> rateWrapped = rate.getFlattenedView();

        int length = ArrayUtil.prod(shape);
        int[] samples = new int[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(rateWrapped.getOrScalar(i), random);
        }

        return IntegerTensor.create(samples, shape);
    }

    private static int sample(double rate, KeanuRandom random) {
        if (rate <= 0.) {
            throw new IllegalArgumentException("Invalid value for rate: " + rate);
        }

        double b = 1.;
        double stopB = Math.exp(-rate);
        int i;

        for (i = 0; b >= stopB; i++) {
            b *= random.nextDouble();
        }

        return i - 1;
    }

    @Override
    public DoubleTensor logProb(IntegerTensor x) {
        Tensor.FlattenedView<Double> rateFlattenedView = rate.getFlattenedView();
        Tensor.FlattenedView<Integer> kFlattenedView = x.getFlattenedView();

        double[] result = new double[(int) x.getLength()];
        for (int i = 0; i < result.length; i++) {
            result[i] = Math.log(pmf(rateFlattenedView.getOrScalar(i), kFlattenedView.getOrScalar(i)));
        }

        return DoubleTensor.create(result, x.getShape());
    }

    private static double pmf(double rate, int x) {
        if (x >= 0 && x < 20) {
            return (Math.pow(rate, x) / factorial(x)) * Math.exp(-rate);
        } else if (x >= 20) {
            double sumOfFactorial = 0;
            for (int i = 1; i <= x; i++) {
                sumOfFactorial += Math.log(i);
            }
            return Math.exp((x * Math.log(rate)) - sumOfFactorial - rate);
        }
        return 0;
    }

}