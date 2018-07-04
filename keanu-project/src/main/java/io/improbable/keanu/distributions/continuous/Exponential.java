package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Exponential {

    private Exponential() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor location, DoubleTensor lambda, KeanuRandom random) {
        return location.minus(random.nextDouble(shape).logInPlace().timesInPlace(lambda));
    }

    public static DoubleTensor logPdf(DoubleTensor location, DoubleTensor lambda, DoubleTensor x) {
        final DoubleTensor negXMinusADivB = x.minus(location).unaryMinusInPlace().divInPlace(lambda);
        final DoubleTensor negXMinusADivBMinusLogB = negXMinusADivB.minusInPlace(lambda.log());
        return negXMinusADivBMinusLogB.setWithMask(x.getLessThanMask(location), Double.NEGATIVE_INFINITY);
    }

    public static Diff dlnPdf(DoubleTensor location, DoubleTensor lambda, DoubleTensor x) {
        final DoubleTensor dPda = lambda.reciprocal();
        final DoubleTensor dPdb = x.minus(location).minusInPlace(lambda).divInPlace(lambda.pow(2));
        return new Diff(dPda, dPdb, dPda.unaryMinus());
    }

    public static class Diff {
        public final DoubleTensor dPdlocation;
        public final DoubleTensor dPdlambda;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPda, DoubleTensor dPdb, DoubleTensor dPdx) {
            this.dPdlocation = dPda;
            this.dPdlambda = dPdb;
            this.dPdx = dPdx;
        }
    }

}
