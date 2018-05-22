package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class TensorLogistic {

    public static DoubleTensor sample(int[] shape, DoubleTensor a, DoubleTensor b, KeanuRandom random) {
        return random.nextDouble(shape).reciprocal().minus(1).log().times(a.minus(b));
    }

    public static DoubleTensor logPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor xMinusAOverB = x.minus(a).div(b);
        final DoubleTensor ln1OverB = b.reciprocal().log();

        return xMinusAOverB.plus(ln1OverB).minusInPlace(xMinusAOverB.exp().plus(1).log().times(2));
    }

    public static Diff dlnPdf(DoubleTensor a, DoubleTensor b, DoubleTensor x) {
        final DoubleTensor expAOverB = a.div(b).exp();
        final DoubleTensor expXOverB = x.div(b).exp();
        final DoubleTensor expPlus = expAOverB.plus(expXOverB);
        final DoubleTensor bTimesExpAOverB = expAOverB.times(b);
        final DoubleTensor bTimesExpXOverB = expXOverB.times(b);

        final DoubleTensor dPda = expXOverB.minus(expAOverB).div(b.times(expPlus));
        final DoubleTensor dPdx = expAOverB.minus(expXOverB).divInPlace(bTimesExpAOverB.plus(bTimesExpXOverB));

        final DoubleTensor numeratorPartOne = a.times(expXOverB).plus(x.times(expAOverB)).plus(a.times(expAOverB.unaryMinus()));
        final DoubleTensor numeratorPartTwo = bTimesExpAOverB.plus(bTimesExpXOverB).minus(x.times(expXOverB));
        final DoubleTensor denominator = b.pow(2).times(expPlus);

        final DoubleTensor dPdb = numeratorPartOne.plus(numeratorPartTwo).div(denominator).unaryMinus();

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
