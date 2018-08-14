package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.continuous.Gaussian.LN_SQRT_2PI;
import static io.improbable.keanu.distributions.dual.Diffs.MU;
import static io.improbable.keanu.distributions.dual.Diffs.SIGMA;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier,
 * ARL-TR-2168 March 2000,
 * 5.1.15 page 28"
 */
public class LogNormal implements ContinuousDistribution {

    private final DoubleTensor location;
    private final DoubleTensor scale;

    /**
     * @param location shifts the distribution
     * @param scale    stretches/shrinks the distribution
     */
    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor scale) {
        return new LogNormal(location, scale);
    }

    private LogNormal(DoubleTensor location, DoubleTensor scale) {
        this.location = location;
        this.scale = scale;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return Gaussian.withParameters(location, scale).sample(shape, random).expInPlace();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnStandardDeviationX = scale.times(x).logInPlace();
        final DoubleTensor lnXMinusMuSquared = x.log().minusInPlace(location).powInPlace(2);
        final DoubleTensor lnXMinusMuSquaredOver2Sigma = lnXMinusMuSquared.divInPlace(scale.pow(2).timesInPlace(2.0));
        return lnXMinusMuSquaredOver2Sigma.plusInPlace(lnStandardDeviationX).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor variance = scale.pow(2);
        final DoubleTensor lnXMinusLocation = x.log().minusInPlace(location);

        final DoubleTensor dLogPdlocation = lnXMinusLocation.div(variance);
        final DoubleTensor dLogPdscale = lnXMinusLocation.powInPlace(2)
            .divInPlace(variance.timesInPlace(scale))
            .minusInPlace(scale.reciprocal());
        final DoubleTensor dLogPdx = dLogPdlocation.plus(1.0).unaryMinus().divInPlace(x);

        return new Diffs()
            .put(MU, dLogPdlocation)
            .put(SIGMA, dLogPdscale)
            .put(X, dLogPdx);
    }

}