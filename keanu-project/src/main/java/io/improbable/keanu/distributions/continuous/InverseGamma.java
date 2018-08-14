package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.Diffs.A;
import static io.improbable.keanu.distributions.dual.Diffs.B;
import static io.improbable.keanu.distributions.dual.Diffs.X;

import org.apache.commons.math3.special.Gamma;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class InverseGamma implements ContinuousDistribution {

    private final DoubleTensor alpha;
    private final DoubleTensor scale;

    /**
     * <h3>Inverted Gamma (Person's Type 6) Distribution</h3>
     *
     * @param alpha    shape parameter (not to be confused with tensor shape)
     * @param scale    stretches/shrinks the distribution
     * @see "Computer Generation of Statistical Distributions
     * by Richard Saucier,
     * ARL-TR-2168 March 2000,
     * 5.1.4 page 16"
     */
    public static ContinuousDistribution withParameters(DoubleTensor alpha, DoubleTensor scale) {
        return new InverseGamma(alpha, scale);
    }

    private InverseGamma(DoubleTensor alpha, DoubleTensor scale) {
        this.alpha = alpha;
        this.scale = scale;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        final DoubleTensor gammaSample = random.nextGamma(shape, scale.reciprocal(), alpha);
        return gammaSample.reciprocal();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor aTimesLnB = alpha.times(scale.log());
        final DoubleTensor negAMinus1TimesLnX = x.log().timesInPlace(alpha.unaryMinus().minusInPlace(1));
        final DoubleTensor lnGammaA = alpha.apply(Gamma::gamma).logInPlace();

        return aTimesLnB.plus(negAMinus1TimesLnX).minusInPlace(lnGammaA).minusInPlace(scale.div(x));
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dLogPdalpha = x.log().unaryMinusInPlace().minusInPlace(alpha.apply(Gamma::digamma)).plusInPlace(scale.log());
        final DoubleTensor dLogPdscale = x.reciprocal().unaryMinusInPlace().plusInPlace(alpha.div(scale));
        final DoubleTensor dLogPdx = x.pow(2).reciprocalInPlace().timesInPlace(x.times(alpha.plus(1).unaryMinusInPlace()).plusInPlace(scale));

        return new Diffs()
            .put(A, dLogPdalpha)
            .put(B, dLogPdscale)
            .put(X, dLogPdx);
    }

}