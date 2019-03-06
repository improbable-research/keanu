package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.hyperparam.Diffs;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.DoublePlaceholderVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import static io.improbable.keanu.distributions.hyperparam.Diffs.A;
import static io.improbable.keanu.distributions.hyperparam.Diffs.B;
import static io.improbable.keanu.distributions.hyperparam.Diffs.X;

public class InverseGamma implements ContinuousDistribution {

    private final DoubleTensor alpha;
    private final DoubleTensor beta;

    public static ContinuousDistribution withParameters(DoubleTensor alpha, DoubleTensor beta) {
        return new InverseGamma(alpha, beta);
    }

    private InverseGamma(DoubleTensor alpha, DoubleTensor beta) {
        this.alpha = alpha;
        this.beta = beta;
    }

    @Override
    public DoubleTensor sample(long[] shape, KeanuRandom random) {
        final DoubleTensor gammaSample = random.nextGamma(shape, beta.reciprocal(), alpha);
        return gammaSample.reciprocal();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor aTimesLnB = alpha.times(beta.log());
        final DoubleTensor negAMinus1TimesLnX = x.log().timesInPlace(alpha.unaryMinus().minusInPlace(1.));
        final DoubleTensor lnGammaA = alpha.logGamma();

        return aTimesLnB.plus(negAMinus1TimesLnX).minusInPlace(lnGammaA).minusInPlace(beta.div(x));
    }

    public static DoubleVertex logProbOutput(DoublePlaceholderVertex x, DoublePlaceholderVertex alpha, DoublePlaceholderVertex beta) {
        final DoubleVertex aTimesLnB = alpha.times(beta.log());
        final DoubleVertex negAMinus1TimesLnX = x.log().times(alpha.unaryMinus().minus(1.));
        final DoubleVertex lnGammaA = alpha.logGamma();

        return aTimesLnB.plus(negAMinus1TimesLnX).minus(lnGammaA).minus(beta.div(x));
    }

    @Override
    public Diffs dLogProb(DoubleTensor x) {
        final DoubleTensor dPdalpha = x.log().unaryMinusInPlace().minusInPlace(alpha.digamma()).plusInPlace(beta.log());
        final DoubleTensor dLogPdbeta = x.reciprocal().unaryMinusInPlace().plusInPlace(alpha.div(beta));
        final DoubleTensor dLogPdx = x.pow(2).reciprocalInPlace().timesInPlace(x.times(alpha.plus(1).unaryMinusInPlace()).plusInPlace(beta));

        return new Diffs()
            .put(A, dPdalpha)
            .put(B, dLogPdbeta)
            .put(X, dLogPdx);
    }

}
