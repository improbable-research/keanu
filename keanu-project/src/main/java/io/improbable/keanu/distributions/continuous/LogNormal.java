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

    private final DoubleTensor scale;
    private final DoubleTensor distributionShape;

    /**
     * @param scale             stretches/shrinks the distribution
     * @param distributionShape shape parameter (not to be confused with tensor shape)
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor scale, DoubleTensor distributionShape) {
        return new LogNormal(scale, distributionShape);
    }

    private LogNormal(DoubleTensor scale, DoubleTensor distributionShape) {
        this.scale = scale;
        this.distributionShape = distributionShape;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        return Gaussian.withParameters(scale, distributionShape).sample(shape, random).expInPlace();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor lnSigmaX = distributionShape.times(x).logInPlace();
        final DoubleTensor lnXMinusMuSquared = x.log().minusInPlace(scale).powInPlace(2);
        final DoubleTensor lnXMinusMuSquaredOver2Variance = lnXMinusMuSquared.divInPlace(distributionShape.pow(2).timesInPlace(2.0));
        return lnXMinusMuSquaredOver2Variance.plusInPlace(lnSigmaX).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor variance = distributionShape.pow(2);
        final DoubleTensor lnXMinusMu = x.log().minusInPlace(scale);

        final DoubleTensor dLogPdmu = lnXMinusMu.div(variance);
        final DoubleTensor dLogPdx = dLogPdmu.plus(1.0).unaryMinus().divInPlace(x);
        final DoubleTensor dLogPdsigma = lnXMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(distributionShape))
            .minusInPlace(distributionShape.reciprocal());

        return new Diffs()
            .put(MU, dLogPdmu)
            .put(SIGMA, dLogPdsigma)
            .put(X, dLogPdx);
    }

}