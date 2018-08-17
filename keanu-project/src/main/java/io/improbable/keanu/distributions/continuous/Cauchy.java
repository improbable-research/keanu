package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.dual.Diffs.L;
import static io.improbable.keanu.distributions.dual.Diffs.S;
import static io.improbable.keanu.distributions.dual.Diffs.X;

public class Cauchy implements ContinuousDistribution {

    private static final double NEG_LOG_PI = -Math.log(Math.PI);
    private final DoubleTensor location;
    private final DoubleTensor scale;

    /**
     * <h3>Cauchy (Lorentz) Distribution</h3>
     *
     * @param location shifts the distribution
     * @param scale    stretches/shrinks the distribution, must be greater than 0
     * @see "Computer Generation of Statistical Distributions
     * by Richard Saucier,
     * ARL-TR-2168 March 2000,
     * 5.1.3 page 15"
     */
    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor scale) {
        return new Cauchy(location, scale);
    }

    private Cauchy(DoubleTensor location, DoubleTensor scale) {
        this.location = location;
        this.scale = scale;
    }

    /**
     * @throws IllegalArgumentException if non-positive scale was passed to {@link #withParameters(DoubleTensor location, DoubleTensor scale)}
     */
    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        if (!scale.greaterThan(0.0).allTrue()) {
            throw new IllegalArgumentException("Invalid argument for Scale. It must be greater than 0. Scale: " + scale.scalar());
        }
        DoubleTensor unityCauchy = random.nextDouble(shape);
        return unityCauchy.minusInPlace(0.5).timesInPlace(Math.PI).tanInPlace().timesInPlace(scale).plusInPlace(location);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor negLnScaleMinusLnPi = scale.log().unaryMinusInPlace().plusInPlace(NEG_LOG_PI);
        final DoubleTensor xMinusLocationOverScalePow2Plus1 = x.minus(location).divInPlace(scale).powInPlace(2).plusInPlace(1);
        final DoubleTensor lnXMinusLocationOverScalePow2Plus1 = xMinusLocationOverScalePow2Plus1.logInPlace();

        return negLnScaleMinusLnPi.minusInPlace(lnXMinusLocationOverScalePow2Plus1);
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor xMinusLocation = x.minus(location);
        final DoubleTensor xMinusLocationPow2 = xMinusLocation.pow(2);
        final DoubleTensor scalePow2 = scale.pow(2);
        final DoubleTensor locationTimesXTimes2 = location.times(x).timesInPlace(2);

        final DoubleTensor dLogPdlocation = xMinusLocation.times(2).divInPlace(scalePow2.plus(xMinusLocationPow2));
        final DoubleTensor dLogPdscale = xMinusLocationPow2.minus(scalePow2).divInPlace(scale.times(xMinusLocationPow2.plus(scalePow2)));

        final DoubleTensor dLogPdxDenominator = location.pow(2).minusInPlace(locationTimesXTimes2).plusInPlace(scalePow2).plusInPlace(x.pow(2));
        final DoubleTensor dLogPdx = xMinusLocation.times(-2).divInPlace(dLogPdxDenominator);

        return new Diffs()
            .put(L, dLogPdlocation)
            .put(S, dLogPdscale)
            .put(X, dLogPdx);
    }

}