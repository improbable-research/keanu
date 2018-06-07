package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import org.apache.commons.math3.special.Gamma;

public class TensorInverseGamma {

    private TensorInverseGamma() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor a, DoubleTensor b, KeanuRandom random) {
        final DoubleTensor gammaSample = random.nextGamma(shape, DoubleTensor.ZERO_SCALAR, b.reciprocal(), a);
        return gammaSample.reciprocal();
    }

    public static DoubleTensor logPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor aTimesLnB = a.times(b.log());
        final DoubleTensor negAMinus1TimesLnX = x.log().timesInPlace(a.unaryMinus().minusInPlace(1));
        final DoubleTensor lnGammaA = a.apply(Gamma::gamma).logInPlace();

        return aTimesLnB.plus(negAMinus1TimesLnX).minusInPlace(lnGammaA).minusInPlace(b.div(x));
    }

    public static Diff dlnPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor dPda = x.log().unaryMinusInPlace().minusInPlace(a.apply(Gamma::digamma)).plusInPlace(b.log());
        final DoubleTensor dPdb = x.reciprocal().unaryMinusInPlace().plusInPlace(a.div(b));
        final DoubleTensor dPdx = x.pow(2).reciprocalInPlace().timesInPlace(x.times(a.plus(1).unaryMinusInPlace()).plusInPlace(b));

        return new Diff(dPda, dPdb, dPdx);
    }

    public static class Diff {

        public final DoubleTensor dPda;
        public final DoubleTensor dPdb;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPda, DoubleTensor dPdb, DoubleTensor dPdx) {
            this.dPda = dPda;
            this.dPdb = dPdb;
            this.dPdx = dPdx;
        }

    }

}
