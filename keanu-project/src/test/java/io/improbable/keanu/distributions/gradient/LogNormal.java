package io.improbable.keanu.distributions.gradient;

public class LogNormal {

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

