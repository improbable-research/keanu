package io.improbable.keanu.distributions.tensors.continuous;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class NDGaussian {

    private static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    private static final double LN_SQRT_2PI = Math.log(SQRT_2PI);

    private NDGaussian() {
    }

    public static DoubleTensor sample(DoubleTensor mu, DoubleTensor sigma, KeanuRandom random) {
        DoubleTensor unityGaussian = random.nextGaussian(mu.getShape());
        return unityGaussian.times(sigma).plus(mu);
    }

    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {
        final DoubleTensor lnSigma = sigma.log();
        final DoubleTensor xMinusMu = x.minus(mu);
        final DoubleTensor xMinusMuOver2Variance = xMinusMu.times(xMinusMu).div(sigma.times(sigma).times(2.0));
        return xMinusMuOver2Variance.minus(lnSigma).minus(LN_SQRT_2PI).unaryMinus();
    }

    public static Diff dlnPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {
        final DoubleTensor variance = sigma.times(sigma);
        final DoubleTensor xMinusMu = x.minus(mu);

        final DoubleTensor dlnP_dmu = xMinusMu.div(variance);
        final DoubleTensor dlnP_dx = dlnP_dmu.unaryMinus();
        final DoubleTensor dlnP_dsigma = ((xMinusMu.times(xMinusMu)).div(variance.times(sigma))).minus(sigma.reciprocal());

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
