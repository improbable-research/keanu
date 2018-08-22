package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.distributions.BaseDistribution;
import io.improbable.keanu.distributions.IntegerSupport;
import io.improbable.keanu.distributions.Support;
import io.improbable.keanu.tensor.intgr.Nd4jIntegerTensor;
import static org.apache.commons.math3.util.CombinatoricsUtils.factorial;

import org.nd4j.linalg.util.ArrayUtil;

import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.2.8 page 49
 */
public class Poisson implements DiscreteDistribution {

    private final DoubleTensor mu;

    public static DiscreteDistribution withParameters(DoubleTensor mu) {
        return new Poisson(mu);
    }

    private Poisson(DoubleTensor mu) {
        this.mu = mu;
    }

    @Override
    public IntegerTensor sample(int[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> muWrapped = mu.getFlattenedView();

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
        Tensor.FlattenedView<Double> muFlattenedView = mu.getFlattenedView();
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

    @Override
    public Support<IntegerTensor> getSupport() {
        return new IntegerSupport(
            Nd4jIntegerTensor.zeros(mu.getShape()),
            Nd4jIntegerTensor.create(Integer.MAX_VALUE, mu.getShape()),
            mu.getShape());
    }

    @Override
    public DoubleTensor computeKLDivergence(BaseDistribution q) {
        if (q instanceof Poisson) {
            DoubleTensor qMu = ((Poisson) q).mu;
            return mu.times(mu.div(qMu).logInPlace()).plus(qMu).minus(mu);
        } else {
            return DiscreteDistribution.super.computeKLDivergence(q);
        }
    }
}
