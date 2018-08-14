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

    public static Diff dlnPdf(double lambda, double x) {
        double dPdlambda = -(lambda - x) / Math.pow(lambda, 2);
        double dPdx = -(1 / lambda);
        return new Diff(dPdlambda, dPdx);
    }

    public static class Diff {
        public final double dPdlambda;
        public final double dPdx;

        public Diff(double dPdlambda, double dPdx) {
            this.dPdlambda = dPdlambda;
            this.dPdx = dPdx;
        }
    }
}
