package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.A;
import static io.improbable.keanu.distributions.dual.Diffs.B;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import org.apache.commons.math3.special.Gamma;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * @see "Computer Generation of Statistical Distributions
 * by Richard Saucier,
 * ARL-TR-2168 March 2000,
 * 5.1.4 page 16"
 */
public class InverseGamma implements ContinuousDistribution {

    private final DoubleTensor distributionShape;
    private final DoubleTensor scale;

    /**
     * @param distributionShape shape parameter (not to be confused with tensor shape)
     * @param scale             stretches/shrinks the distribution
     * @return an instance of {@link ContinuousDistribution}
     */
    public static ContinuousDistribution withParameters(DoubleTensor distributionShape, DoubleTensor scale) {
        return new InverseGamma(distributionShape, scale);
    }

    private InverseGamma(DoubleTensor distributionShape, DoubleTensor scale) {
        this.distributionShape = distributionShape;
        this.scale = scale;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        final DoubleTensor gammaSample = random.nextGamma(shape, scale.reciprocal(), distributionShape);
        return gammaSample.reciprocal();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor aTimesLnB = distributionShape.times(scale.log());
        final DoubleTensor negAMinus1TimesLnX = x.log().timesInPlace(distributionShape.unaryMinus().minusInPlace(1));
        final DoubleTensor lnGammaA = distributionShape.apply(Gamma::gamma).logInPlace();

        return aTimesLnB.plus(negAMinus1TimesLnX).minusInPlace(lnGammaA).minusInPlace(scale.div(x));
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdalpha = x.log().unaryMinusInPlace().minusInPlace(distributionShape.apply(Gamma::digamma)).plusInPlace(scale.log());
        final DoubleTensor dLogPdscale = x.reciprocal().unaryMinusInPlace().plusInPlace(distributionShape.div(scale));
        final DoubleTensor dLogPdx = x.pow(2).reciprocalInPlace().timesInPlace(x.times(distributionShape.plus(1).unaryMinusInPlace()).plusInPlace(scale));

        return new Diffs()
            .put(A, dLogPdalpha)
            .put(B, dLogPdscale)
            .put(X, dLogPdx);
    }

}