package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.special.Gamma;

public class InverseGamma {

    private InverseGamma() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor alpha, DoubleTensor beta, KeanuRandom random) {
        final DoubleTensor gammaSample = random.nextGamma(shape, DoubleTensor.ZERO_SCALAR, beta.reciprocal(), alpha);
        return gammaSample.reciprocal();
    }

    public static DoubleTensor logPdf(DoubleTensor alpha, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor aTimesLnB = alpha.times(beta.log());
        final DoubleTensor negAMinus1TimesLnX = x.log().timesInPlace(alpha.unaryMinus().minusInPlace(1));
        final DoubleTensor lnGammaA = alpha.apply(Gamma::gamma).logInPlace();

        return aTimesLnB.plus(negAMinus1TimesLnX).minusInPlace(lnGammaA).minusInPlace(beta.div(x));
    }

    public static Diff dlnPdf(DoubleTensor alpha, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor dPda = x.log().unaryMinusInPlace().minusInPlace(alpha.apply(Gamma::digamma)).plusInPlace(beta.log());
        final DoubleTensor dPdb = x.reciprocal().unaryMinusInPlace().plusInPlace(alpha.div(beta));
        final DoubleTensor dPdx = x.pow(2).reciprocalInPlace().timesInPlace(x.times(alpha.plus(1).unaryMinusInPlace()).plusInPlace(beta));

        return new Diff(dPda, dPdb, dPdx);
    }

    public static class Diff {

        public final DoubleTensor dPdalpha;
        public final DoubleTensor dPdbeta;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPdalpha, DoubleTensor dPdbeta, DoubleTensor dPdx) {
            this.dPdalpha = dPdalpha;
            this.dPdbeta = dPdbeta;
            this.dPdx = dPdx;
        }

    }

}
