package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.continuous.Gaussian.LN_SQRT_2PI;

public class LogNormal {

    private LogNormal() {
    }

    /**
     * @param shape  shape of tensor returned
     * @param mu     location parameter (any real number)
     * @param sigma  square root of variance (greater than 0)
     * @param random source or randomness
     * @return a sample from the distribution given mu and sigma
     */
    public static DoubleTensor sample(int[] shape, DoubleTensor mu, DoubleTensor sigma, KeanuRandom random) {
        return Gaussian.sample(shape, mu, sigma, random).expInPlace();
    }

    /**
     * @param mu    location parameter (any real number)
     * @param sigma square root of variance (greater than 0)
     * @param x     at value
     * @return the natural log of the pdf given mu and sigma at x
     */
    public static DoubleTensor logPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {
        final DoubleTensor lnSigmaX = sigma.times(x).logInPlace();
        final DoubleTensor lnXMinusMuSquared = x.log().minusInPlace(mu).powInPlace(2);
        final DoubleTensor lnXMinusMuSquaredOver2Variance = lnXMinusMuSquared.divInPlace(sigma.pow(2).timesInPlace(2.0));
        return lnXMinusMuSquaredOver2Variance.plusInPlace(lnSigmaX).plusInPlace(LN_SQRT_2PI).unaryMinusInPlace();
    }

    public static LogNormal.Diff dlnPdf(DoubleTensor mu, DoubleTensor sigma, DoubleTensor x) {
        final DoubleTensor variance = sigma.pow(2);
        final DoubleTensor lnXMinusMu = x.log().minusInPlace(mu);

        final DoubleTensor dlnP_dmu = lnXMinusMu.div(variance);
        final DoubleTensor dlnP_dx = dlnP_dmu.plus(1.0).unaryMinus().divInPlace(x);
        final DoubleTensor dlnP_dsigma = lnXMinusMu.powInPlace(2)
            .divInPlace(variance.timesInPlace(sigma))
            .minusInPlace(sigma.reciprocal());

        return new LogNormal.Diff(dlnP_dmu, dlnP_dsigma, dlnP_dx);
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
