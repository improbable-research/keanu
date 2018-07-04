package io.improbable.keanu.distributions.gradient;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.14 page 27
 */
public class Logistic {

    private Logistic() {
    }

    public static Diff dlnPdf(double a, double b, double x) {
        double expAOverB = Math.exp(a / b);
        double expXOverB = Math.exp(x / b);

        double dPda = (expXOverB - expAOverB) / (b * (expAOverB + expXOverB));
        double dPdx = (expAOverB - expXOverB) / ((b * expAOverB) + (b * expXOverB));
        double dPdb = -(((a * expXOverB) + (x * expAOverB) + (a * -expAOverB) + (b * expAOverB) + (b * expXOverB) - (x * expXOverB)) /
            (b * b * (expAOverB + expXOverB)));
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
