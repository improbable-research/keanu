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
     * @param rate rate of occurrence, must be greater than 0
     * @return an instance of {@link DiscreteDistribution}
     */
    public static DiscreteDistribution withParameters(DoubleTensor rate) {
        return new Poisson(rate);
    }

    private Poisson(DoubleTensor rate) {
        this.rate = rate;
    }

     /**
     * @param shape  an integer array describing the shape of the tensors to be sampled
     * @param random {@link KeanuRandom}
     * @return an instance of {@link DoubleTensor}
     * @throws IllegalArgumentException if <code>rate</code> passed to {@link #withParameters(DoubleTensor rate)} is less than or equal to 0
     */
     @Override
     public IntegerTensor sample(int[] shape, KeanuRandom random) {
         Tensor.FlattenedView<Double> muWrapped = rate.getFlattenedView();

         int length = ArrayUtil.prod(shape);
         int[] samples = new int[length];
         for (int i = 0; i < length; i++) {
             samples[i] = sample(muWrapped.getOrScalar(i), random);
         }

         return IntegerTensor.create(samples, shape);
     }

    private static int sample(double mu, KeanuRandom random) {
        if (mu <= 0.) {
            throw new IllegalArgumentException("Invalid value for mu: " + mu);
        }

        double b = 1.;
        double stopB = Math.exp(-mu);
        int i;

        for (i = 0; b >= stopB; i++) {
            b *= random.nextDouble();
        }

        return i - 1;
    }

    @Override
    public DoubleTensor logProb(IntegerTensor k) {
        Tensor.FlattenedView<Double> muFlattenedView = rate.getFlattenedView();
        Tensor.FlattenedView<Integer> kFlattenedView = k.getFlattenedView();

        double[] result = new double[(int) k.getLength()];
        for (int i = 0; i < result.length; i++) {
            result[i] = Math.log(pmf(muFlattenedView.getOrScalar(i), kFlattenedView.getOrScalar(i)));
        }

        return DoubleTensor.create(result, k.getShape());
    }

    private static double pmf(double mu, int k) {
        if (k >= 0 && k < 20) {
            return (Math.pow(mu, k) / factorial(k)) * Math.exp(-mu);
        } else if (k >= 20) {
            double sumOfFactorial = 0;
            for (int i = 1; i <= k; i++) {
                sumOfFactorial += Math.log(i);
            }
            return Math.exp((k * Math.log(mu)) - sumOfFactorial - mu);
        }
        return 0;
    }

}