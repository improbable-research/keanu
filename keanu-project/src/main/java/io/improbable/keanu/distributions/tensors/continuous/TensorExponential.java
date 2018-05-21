package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class TensorExponential {

    private TensorExponential() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor a, DoubleTensor b, KeanuRandom random) {
        return a.minus(b).times(random.nextDouble(shape).logInPlace());
    }

    public static DoubleTensor logPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor negXMinusADivB = x.minus(a).unaryMinus().divInPlace(b);
        final DoubleTensor negXMinusADivBMinusLogB = negXMinusADivB.minusInPlace(b.log());
        return negXMinusADivBMinusLogB.applyWhere(x.getLessThanMask(a), 0.0);
    }

    public static Diff dlnPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor dPda = b.reciprocal();
        final DoubleTensor dPdb = x.minus(a).minus(b).div(b.pow(2));
        return new Diff(dPda, dPdb, dPda.unaryMinus());
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
