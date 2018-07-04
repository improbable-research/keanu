package io.improbable.keanu.distributions.gradient;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 20
 */

public class Exponential {

    private Exponential() {
    }

    public static Diff dlnPdf(double a, double b, double x) {
        double dPda = 1 / b;
        double dPdb = -(a + b - x) / Math.pow(b, 2);
        double dPdx = -dPda;
        return new Diff(dPda, dPdb, dPdx);
    }

    public static class Diff {
        public final double dPda;
        public final double dPdb;
        public final double dPdx;

        public Diff(double dPda, double dPdb, double dPdx) {
            this.dPda = dPda;
            this.dPdb = dPdb;
            this.dPdx = dPdx;
        }
    }
}
