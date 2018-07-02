package io.improbable.keanu.distributions.gradient;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 25
 */

public class Laplace {

    private Laplace() {
    }

    public static Diff dlnPdf(double mu, double beta, double x) {
        double denominator = (beta * Math.abs(mu - x));

        double dPdx = (mu - x) / denominator;
        double dPdm = (x - mu) / denominator;
        double dPdb = (Math.abs(mu - x) - beta) / Math.pow(beta, 2);
        return new Diff(dPdm, dPdb, dPdx);
    }

    public static class Diff {
        public final double dPdmu;
        public final double dPdbeta;
        public final double dPdx;

        public Diff(double dPdmu, double dPdbeta, double dPdx) {
            this.dPdmu = dPdmu;
            this.dPdbeta = dPdbeta;
            this.dPdx = dPdx;
        }
    }
}
