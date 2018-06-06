package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class TensorExponential {

    private TensorExponential() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor a, DoubleTensor b, KeanuRandom random) {
        return a.minus(random.nextDouble(shape).logInPlace().timesInPlace(b));
    }

    public static DoubleTensor logPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor negXMinusADivB = x.minus(a).unaryMinusInPlace().divInPlace(b);
        final DoubleTensor negXMinusADivBMinusLogB = negXMinusADivB.minusInPlace(b.log());
        return negXMinusADivBMinusLogB.setWithMask(x.getLessThanMask(a), Double.NEGATIVE_INFINITY);
    }

    public static Diff dlnPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor dPda = b.reciprocal();
        final DoubleTensor dPdb = x.minus(a).minusInPlace(b).divInPlace(b.pow(2));
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
