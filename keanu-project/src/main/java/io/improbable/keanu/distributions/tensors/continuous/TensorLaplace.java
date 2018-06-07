package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class TensorLaplace {

    private TensorLaplace() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor beta, KeanuRandom random) {
        return random.nextLaplace(shape, mu, beta);
    }

    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor muMinusXAbsNegDivBeta = mu.minus(x).abs().divInPlace(beta);
        final DoubleTensor logTwoBeta = beta.times(2).logInPlace();
        return muMinusXAbsNegDivBeta.plusInPlace(logTwoBeta).unaryMinus();
    }

    public static Diff dlnPdf(DoubleTensor mu, DoubleTensor beta, DoubleTensor x) {
        final DoubleTensor muMinusX = mu.minus(x);
        final DoubleTensor muMinusXAbs = muMinusX.abs();

        final DoubleTensor denominator = muMinusXAbs.times(beta);

        final DoubleTensor dPdx = muMinusX.divInPlace(denominator);
        final DoubleTensor dPdMu = x.minus(mu).divInPlace(denominator);
        final DoubleTensor dPdBeta = muMinusXAbs.minusInPlace(beta).divInPlace(beta.pow(2));

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
