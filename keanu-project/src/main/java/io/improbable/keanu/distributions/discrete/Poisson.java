package io.improbable.keanu.distributions.discrete;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.DiscreteDistribution;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerPlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.2.8 page 49
 */
public class Poisson implements DiscreteDistribution {

    private final DoubleTensor mu;

    public static Poisson withParameters(DoubleTensor mu) {
        return new Poisson(mu);
    }

    private Poisson(DoubleTensor mu) {
        this.mu = mu;
    }

    @Override
    public IntegerTensor sample(long[] shape, KeanuRandom random) {
        Tensor.FlattenedView<Double> muWrapped = mu.getFlattenedView();

        long muLength = muWrapped.size();
        int length = TensorShape.getLengthAsInt(shape);
        int[] samples = new int[length];
        for (int i = 0; i < length; i++) {
            samples[i] = sample(muWrapped.get(i % muLength), random);
        }

        return IntegerTensor.create(samples, shape);
    }

    private static int sample(double mu, KeanuRandom random) {
        if (mu <= 0.) {
            throw new IllegalArgumentException("Invalid value for mu: " + mu);
        }

        final double STEP_IN_MU = 500;
        double muLeft = mu;
        int k = 0;
        double p = 1.0;

        /*
         * Algorithm courtesy of Wikipedia:
         * https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
         *
         * Designed to introduce mu gradually to avoid numerical stability issues.
         */
        do {
            k++;
            double u = random.nextDoubleNonZero();
            p *= u;

            while (p < 1.0 && muLeft > 0.0) {
                if (muLeft > STEP_IN_MU) {
                    p *= Math.exp(STEP_IN_MU);
                    muLeft -= STEP_IN_MU;
                } else {
                    p *= Math.exp(muLeft);
                    muLeft = 0.0;
                }
            }
        } while (p > 1.0);

        return k - 1;
    }

    @Override
    public DoubleTensor logProb(IntegerTensor k) {

        DoubleTensor kDouble = k.toDouble();
        DoubleTensor logFactorialK = kDouble.plus(1).logGammaInPlace();

        return kDouble.timesInPlace(mu.log()).minusInPlace(mu).minusInPlace(logFactorialK);
    }

    public static DoubleVertex logProbOutput(IntegerPlaceholderVertex k, DoublePlaceholderVertex mu) {

        DoubleVertex kDouble = k.toDouble();
        DoubleVertex logFactorialK = kDouble.plus(1).logGamma();

        return kDouble.times(mu.log()).minus(mu).minus(logFactorialK);
    }

    public DoubleTensor[] dLogProb(IntegerTensor k, boolean wrtMu) {
        DoubleTensor[] result = new DoubleTensor[1];

        if (wrtMu) {
            result[0] = k.toDouble().div(mu).minusInPlace(1.0);
        }

        return result;
    }
}
