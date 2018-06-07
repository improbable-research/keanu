package io.improbable.keanu.distributions.continuous;

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

    /**
     * @param a      location
     * @param b      shape
     * @param random source of randomness
     * @return a random number from the Exponential distribution
     */
    public static double sample(double a, double b, KeanuRandom random) {
        if (b <= 0.0) {
            throw new IllegalArgumentException("Invalid value for b");
        }
        return a - b * Math.log(random.nextDouble());
    }

    public static double pdf(double a, double b, double x) {
        return (x >= a) ? (1.0 / b) * Math.exp(-(x - a) / b) : 0.0;
    }

    public static double logPdf(double a, double b, double x) {
        return (x >= a) ? (-(x - a) / b) - Math.log(b) : Double.NEGATIVE_INFINITY;
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
