package io.improbable.keanu.distributions.continuous;

import static io.improbable.keanu.distributions.dual.ParameterName.A;
import static io.improbable.keanu.distributions.dual.ParameterName.B;
import static io.improbable.keanu.distributions.dual.ParameterName.X;

import org.apache.commons.math3.special.Gamma;

import io.improbable.keanu.distributions.ContinuousDistribution;
import io.improbable.keanu.distributions.dual.ParameterMap;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class InverseGamma implements ContinuousDistribution {

    private final DoubleTensor alpha;
    private final DoubleTensor beta;

    // package private
    InverseGamma(DoubleTensor alpha, DoubleTensor beta) {
        this.alpha = alpha;
        this.beta = beta;
    }

    @Override
    public DoubleTensor sample(int[] shape, KeanuRandom random) {
        final DoubleTensor gammaSample = random.nextGamma(shape, DoubleTensor.ZERO_SCALAR, beta.reciprocal(), alpha);
        return gammaSample.reciprocal();
    }

    @Override
    public DoubleTensor logProb(DoubleTensor x) {
        final DoubleTensor aTimesLnB = alpha.times(beta.log());
        final DoubleTensor negAMinus1TimesLnX = x.log().timesInPlace(alpha.unaryMinus().minusInPlace(1));
        final DoubleTensor lnGammaA = alpha.apply(Gamma::gamma).logInPlace();

        return aTimesLnB.plus(negAMinus1TimesLnX).minusInPlace(lnGammaA).minusInPlace(beta.div(x));
    }

    @Override
    public ParameterMap<DoubleTensor> dLogProb(DoubleTensor x) {
        final DoubleTensor dPdalpha = x.log().unaryMinusInPlace().minusInPlace(alpha.apply(Gamma::digamma)).plusInPlace(beta.log());
        final DoubleTensor dLogPdbeta = x.reciprocal().unaryMinusInPlace().plusInPlace(alpha.div(beta));
        final DoubleTensor dLogPdx = x.pow(2).reciprocalInPlace().timesInPlace(x.times(alpha.plus(1).unaryMinusInPlace()).plusInPlace(beta));

        return new ParameterMap<DoubleTensor>()
            .put(A, dPdalpha)
            .put(B, dLogPdbeta)
            .put(X, dLogPdx);
    }

}
