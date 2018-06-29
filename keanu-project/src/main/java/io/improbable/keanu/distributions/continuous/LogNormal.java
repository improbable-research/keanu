package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static io.improbable.keanu.distributions.continuous.Gaussian.LN_SQRT_2PI;


public class LogNormal {

    public static double sample(double mu, double sigma, KeanuRandom random) {
        return Math.exp(Gaussian.sample(mu, sigma, random));
    }

    public static double logPdf(double mu, double sigma, double x) {
        final double lnSigmaX = Math.log(sigma * x);
        final double lnXMinusMu = Math.log(x) - mu;
        final double lnXMinusMuOver2Variance = lnXMinusMu * lnXMinusMu / (2.0 * sigma * sigma);
        return -lnXMinusMuOver2Variance - lnSigmaX - LN_SQRT_2PI;
    }

    public static LogNormal.Diff dlnPdf(double mu, double sigma, double x) {
        final double variance = sigma * sigma;
        final double lnXMinusMu = Math.log(x) - mu;

        final double dlnP_dmu = lnXMinusMu / variance;
        final double dlnP_dx = -(dlnP_dmu + 1.0) / x;
        final double dlnP_dsigma = ((lnXMinusMu * lnXMinusMu) / (variance * sigma)) - 1 / sigma;

        return new LogNormal.Diff(dlnP_dmu, dlnP_dsigma, dlnP_dx);
    }

    public static class Diff {
        public final double dPdmu;
        public final double dPdsigma;
        public final double dPdx;

        public Diff(double dPdmu, double dPdsigma, double dPdx) {
            this.dPdmu = dPdmu;
            this.dPdsigma = dPdsigma;
            this.dPdx = dPdx;
        }
    }
}

