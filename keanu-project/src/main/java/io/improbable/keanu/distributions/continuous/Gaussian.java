package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class Gaussian {

    private Gaussian() {
    }

    private static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    private static final double LN_SQRT_2PI = Math.log(SQRT_2PI);

    public static double sample(double mu, double sigma, KeanuRandom random) {
        return random.nextGaussian() * sigma + mu;
    }

    public static double pdf(double mu, double sigma, double x) {
        final double normalizer = 1.0 / (sigma * SQRT_2PI);
        final double xMinusMu = x - mu;
        final double exponent = -(xMinusMu * xMinusMu) / (2.0 * sigma * sigma);
        return normalizer * Math.exp(exponent);
    }

    public static double logPdf(double mu, double sigma, double x) {
        final double lnSigma = Math.log(sigma);
        final double xMinusMu = x - mu;
        final double xMinusMuOver2Variance = xMinusMu * xMinusMu / (2 * sigma * sigma);
        return -xMinusMuOver2Variance - lnSigma - LN_SQRT_2PI;
    }

    public static Diff dlnPdf(double mu, double sigma, double x) {
        final double variance = sigma * sigma;
        final double xMinusMu = x - mu;

        final double dlnP_dmu = xMinusMu / variance;
        final double dlnP_dx = -dlnP_dmu;
        final double dlnP_dsigma = ((xMinusMu * xMinusMu) / (variance * sigma)) - 1 / sigma;

        return new Diff(dlnP_dmu, dlnP_dsigma, dlnP_dx);
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
