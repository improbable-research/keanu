package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.dual.Diffs.LAMBDA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier,
 * ARL-TR-2168 March 2000,
 * 5.1.8 page 20"
 */
public class Exponential implements ContinuousDistribution {

    private final DoubleTensor rate;

    /**
     * @param rate inverse scale
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor rate) {
        return new Exponential(rate);
    }

    private Exponential(DoubleTensor rate) {
        this.rate = rate;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return random.nextDouble(shape).logInPlace().timesInPlace(rate).unaryMinusInPlace();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor negXMinusADivB = x.unaryMinus().divInPlace(rate);
        final DoubleTensor negXMinusADivBMinusLogB = negXMinusADivB.minusInPlace(rate.log());
        return negXMinusADivBMinusLogB.setWithMask(x.getLessThanMask(DoubleTensor.ZERO_SCALAR), Double.NEGATIVE_INFINITY);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdx = rate.reciprocal().unaryMinusInPlace();
        final DoubleTensor dLogPdrate = x.minus(rate).divInPlace(rate.pow(2));
        return new Diffs()
            .put(LAMBDA, dLogPdrate)
            .put(X, dLogPdx);
    }

}