package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Gaussian {

    public static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    public static final double LN_SQRT_2PI = Math.log(SQRT_2PI);

    private Gaussian() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor sigma, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(shape);
        return unityGaussian.timesInPlace(sigma).plusInPlace(mu);
    }

    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {
        final DoubleTensor lnSigma = sigma.log();
        final DoubleTensor xMinusMuSquared = x.minus(mu).powInPlace(2);
        final DoubleTensor xMinusMuSquaredOver2Variance = xMinusMuSquared.divInPlace(sigma.pow(2).timesInPlace(2.0));
        return xMinusMuSquaredOver2Variance.plusInPlace(lnSigma).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    public static DiffLogP dlnPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {
        final DoubleTensor variance = sigma.pow(2);
        final DoubleTensor xMinusMu = x.minus(mu);

        final DoubleTensor dLogPdmu = xMinusMu.div(variance);
        final DoubleTensor dLogPdx = dLogPdmu.unaryMinus();
        final DoubleTensor dLogPdsigma = xMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigma))
            .minusInPlace(sigma.reciprocal());

        return new DiffLogP(dLogPdmu, dLogPdsigma, dLogPdx);
    }

    public static class DiffLogP {
        public final DoubleTensor dLogPdmu;
        public final DoubleTensor dLogPdsigma;
        public final DoubleTensor dLogPdx;

        public DiffLogP(DoubleTensor dLogPdmu, DoubleTensor dLogPdsigma, DoubleTensor dLogPdx) {
            this.dLogPdmu = dLogPdmu;
            this.dLogPdsigma = dLogPdsigma;
            this.dLogPdx = dLogPdx;
        }
    }

}
