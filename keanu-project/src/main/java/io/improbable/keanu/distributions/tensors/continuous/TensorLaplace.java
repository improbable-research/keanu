package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class TensorLaplace {

    private TensorLaplace() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor beta, KeanuRandom random) {
        return random.nextLaplace(shape, mu, beta);
    }

    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor muMinusXAbsNegDivBeta = mu.minus(x).abs().div(beta);
        final DoubleTensor logTwoBeta = beta.times(2).log();
        return muMinusXAbsNegDivBeta.plus(logTwoBeta).unaryMinus();
    }

    public static Diff dlnPdf(DoubleTensor mu, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor muMinusX = x.unaryMinus().plus(mu);
        final DoubleTensor muMinusXAbs = muMinusX.abs();

        final DoubleTensor denominator = muMinusXAbs.times(beta);

        final DoubleTensor dPdx = muMinusX.div(denominator);
        final DoubleTensor dPdMu = x.minus(mu).div(denominator);
        final DoubleTensor dPdBeta = muMinusXAbs.minus(beta).div(beta.pow(2));

        return new Diff(dPdMu, dPdBeta, dPdx);
    }

    public static class Diff {
        public final DoubleTensor dPdmu;
        public final DoubleTensor dPdbeta;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPdmu, DoubleTensor dPdbeta, DoubleTensor dPdx) {
            this.dPdmu = dPdmu;
            this.dPdbeta = dPdbeta;
            this.dPdx = dPdx;
        }
    }

}
