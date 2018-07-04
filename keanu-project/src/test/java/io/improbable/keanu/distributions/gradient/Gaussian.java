package io.improbable.keanu.distributions.gradient;

public class Gaussian {

    private Gaussian() {
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
