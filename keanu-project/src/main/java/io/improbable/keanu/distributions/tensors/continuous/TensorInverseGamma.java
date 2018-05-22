package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;
import io.improbable.keanu.vertices.dbltensor.Nd4jDoubleTensor;
import org.apache.commons.math3.special.Gamma;

public class TensorInverseGamma {

    private TensorInverseGamma(){
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor a, DoubleTensor b, KeanuRandom random) {
        final DoubleTensor gammaSample = random.nextGamma(shape, Nd4jDoubleTensor.zeros(shape), Nd4jDoubleTensor.ones(shape).div(b), a);
        return gammaSample.reciprocal();
    }

    public static DoubleTensor logPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor aTimesLnB = a.times(b.log());
        final DoubleTensor negAMinus1TimesLnX = a.unaryMinus().minus(1).times(x.log());
        final DoubleTensor lnGammaA = a.apply(Gamma::gamma).logInPlace();

        return aTimesLnB.plus(negAMinus1TimesLnX).minus(lnGammaA).minus(b.div(x));
    }

    public static Diff dlnPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor dPda = x.log().unaryMinus().minus(a.apply(Gamma::digamma)).plus(b.log());
        final DoubleTensor dPdb = x.reciprocal().unaryMinus().plus(a.div(b));
        final DoubleTensor dPdx = x.pow(2).reciprocal().times(x.times(a.plus(1).unaryMinus()).plus(b));

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
