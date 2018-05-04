package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class NDGaussian {

    private static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    private static final double LN_SQRT_2PI = Math.log(SQRT_2PI);

    private NDGaussian() {
    }

    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor sigma, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(shape);
        return unityGaussian.times(sigma).plus(mu);
    }

    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {

//        final double lnSigma = Math.log(sigma);
//        final double xMinusMu = x - mu;
//        final double xMinusMuOver2Variance = xMinusMu * xMinusMu / (2 * sigma * sigma);
//        return -xMinusMuOver2Variance - lnSigma - LN_SQRT_2PI;

        final DoubleTensor lnSigma = sigma.log();
        final DoubleTensor xMinusMuSquared = x.minus(mu).powInPlace(2);
        final DoubleTensor xMinusMuSquaredOver2Variance = xMinusMuSquared.divInPlace(sigma.pow(2).timesInPlace(2.0));
        return xMinusMuSquaredOver2Variance.plusInPlace(lnSigma).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    public static Diff dlnPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {

//        final double variance = sigma * sigma;
//        final double xMinusMu = x - mu;
//
//        final double dlnP_dmu = xMinusMu / variance;
//        final double dlnP_dx = -dlnP_dmu;
//        final double dlnP_dsigma = ((xMinusMu * xMinusMu) / (variance * sigma)) - 1 / sigma;
//
//        return new Diff(dlnP_dmu, dlnP_dsigma, dlnP_dx);

        final DoubleTensor variance = sigma.pow(2);
        final DoubleTensor xMinusMu = x.minus(mu);

        final DoubleTensor dlnP_dmu = xMinusMu.div(variance);

        final DoubleTensor dlnP_dx = dlnP_dmu.unaryMinus();

        final DoubleTensor dlnP_dsigma = xMinusMu.powInPlace(2)
                .divInPlace(variance.timesInPlace(sigma))
                .minusInPlace(sigma.reciprocal());

        return new Diff(dlnP_dmu, dlnP_dsigma, dlnP_dx);
    }

    public static class Diff {
        public final DoubleTensor dPdmu;
        public final DoubleTensor dPdsigma;
        public final DoubleTensor dPdx;

        public Diff(DoubleTensor dPdmu, DoubleTensor dPdsigma, DoubleTensor dPdx) {
            this.dPdmu = dPdmu;
            this.dPdsigma = dPdsigma;
            this.dPdx = dPdx;
        }
    }
}
