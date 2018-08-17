package io.improbable.keanu.distributions.continuous;

import org.apache.commons.math3.special.Gamma;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier,
 * ARL-TR-2168 March 2000,
 * 5.1.4 page 16"
 */
public class ChiSquared implements ContinuousDistribution {

    private static final double LOG_TWO = Math.log(2);
    private final IntegerTensor degreesOfFreedom;

    /**
     * @param degreesOfFreedom number of degrees of freedom
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(IntegerTensor degreesOfFreedom) {
        return new ChiSquared(degreesOfFreedom);
    }

    private ChiSquared(IntegerTensor degreesOfFreedom) {
        this.degreesOfFreedom = degreesOfFreedom;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return random.nextGamma(shape, DoubleTensor.TWO_SCALAR, degreesOfFreedom.toDouble().div(2));
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        DoubleTensor halfK = degreesOfFreedom.toDouble().div(2);
        DoubleTensor numerator = halfK.minus(1).timesInPlace(x.log()).minusInPlace(x.div(2));
        DoubleTensor denominator = halfK.times(LOG_TWO).plusInPlace(halfK.apply(Gamma::gamma).logInPlace());
        return numerator.minusInPlace(denominator);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        throw new UnsupportedOperationException();
    }

}