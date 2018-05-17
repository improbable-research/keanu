package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

public class Gaussian {

    private Gaussian() {
    }

    private static final double SQRT_2PI = Math.sqrt(Math.PI * 2);
    private static final double LN_SQRT_2PI = Math.log(SQRT_2PI);

    public static double sample(double mu, double sigma, KeanuRandom random) {
        final double u1 = random.nextDouble();
        final double u2 = random.nextDouble();
        final double w = Math.sqrt(-2.0 * Math.log(u1));
        final double x = 2.0 * Math.PI * u2;
        return w * Math.sin(x) * sigma + mu;
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

    public static Diff dPdf(double mu, double sigma, double x) {
        final double variance = sigma * sigma;
        final double pdf = pdf(mu, variance, x);
        final double xMinusMu = x - mu;

        final double dPdx = -xMinusMu * pdf / variance;
        final double dPdmu = -dPdx;
        final double dPdsigma = (xMinusMu * xMinusMu - variance) * pdf / (variance * sigma);

        return new Diff(dPdmu, dPdsigma, dPdx);
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
