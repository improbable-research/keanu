package io.improbable.keanu.distributions.gradient;

public class Pareto {

    private Pareto() {
    }

    public static Diff dlnPdf(double xm, double alpha, double x) {
        double dPdX = (-1.0 - alpha) / x;
        double dPdXm = alpha / xm;
        double dPdAlpha = 1.0 / alpha + Math.log(xm) - Math.log(x);

        return new Diff(dPdXm, dPdAlpha, dPdX);
    }

    public static class Diff {
        public final double dPdLocation;
        public final double dPdScale;
        public final double dPdX;

        public Diff(double dPdLocation, double dPdScale, double dPdX) {
            this.dPdLocation = dPdLocation;
            this.dPdScale = dPdScale;
            this.dPdX = dPdX;
        }
    }
}
