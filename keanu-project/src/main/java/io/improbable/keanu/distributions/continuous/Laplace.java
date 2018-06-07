package io.improbable.keanu.distributions.continuous;

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

    /**
     * @param mu     location
     * @param beta   shape
     * @param random source of randomness
     * @return a random number from the Laplace distribution
     */
    public static double sample(double mu, double beta, KeanuRandom random) {
        if (beta <= 0.0) {
            throw new IllegalArgumentException("Invalid value for beta: " + beta);
        }
        if (random.nextDouble() > 0.5) {
            return mu + beta * Math.log(random.nextDouble());
        } else {
            return mu - beta * Math.log(random.nextDouble());
        }
    }

    public static double pdf(double mu, double beta, double x) {
        return 1 / (2 * beta) * Math.exp(-Math.abs(x - mu) / beta);
    }

    public static double logPdf(double mu, double beta, double x) {
        return -(Math.abs(mu - x) / beta + Math.log(2 * beta));
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
