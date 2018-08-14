package io.improbable.keanu.distributions.continuous;

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
 * 5.1.16 page 29"
 */
public class Gaussian implements ContinuousDistribution {

    private static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    static final double LN_SQRT_2PI = Math.log(SQRT_2PI);
    private final DoubleTensor location;
    private final DoubleTensor scale;

    /**
     * @param location shifts the distribution; mean
     * @param scale    stretches/shrinks the distribution; variance
     */
    public static ContinuousDistribution withParameters(DoubleTensor location, DoubleTensor scale) {
        return new Gaussian(location, scale);
    }

    private Gaussian(DoubleTensor location, DoubleTensor scale) {
        this.location = location;
        this.scale = scale;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(shape);
        return unityGaussian.timesInPlace(scale).plusInPlace(location);
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnSigma = scale.log();
        final DoubleTensor xMinusMuSquared = x.minus(location).powInPlace(2);
        final DoubleTensor xMinusMuSquaredOver2Variance = xMinusMuSquared.divInPlace(scale.pow(2).timesInPlace(2.0));
        return xMinusMuSquaredOver2Variance.plusInPlace(lnSigma).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor variance = scale.pow(2);
        final DoubleTensor xMinusMu = x.minus(location);

        final DoubleTensor dLogPdlocation = xMinusMu.div(variance);
        final DoubleTensor dLogPdscale = xMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(scale))
            .minusInPlace(scale.reciprocal());
        final DoubleTensor dLogPdx = dLogPdlocation.unaryMinus();

        return new Diffs()
            .put(MU, dLogPdlocation)
            .put(SIGMA, dLogPdscale)
            .put(X, dLogPdx);
    }

}