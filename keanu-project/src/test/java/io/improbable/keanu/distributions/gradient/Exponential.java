package io.improbable.keanu.distributions.gradient;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 20
 */

public class Exponential {

    private Exponential() {
    }

    public static Diff dlnPdf(double b, double x) {
        double dPdb = -(b - x) / Math.pow(b, 2);
        double dPdx = -(1 / b);
        return new Diff(dPdb, dPdx);
    }

    public static class Diff {
        public final double dPdb;
        public final double dPdx;

        public Diff(double dPdb, double dPdx) {
            this.dPdb = dPdb;
            this.dPdx = dPdx;
        }
    }
}
