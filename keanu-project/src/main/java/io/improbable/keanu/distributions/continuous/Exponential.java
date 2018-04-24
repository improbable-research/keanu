package io.improbable.keanu.distributions.continuous;

import java.util.Random;

public class Exponential {

    /**
     * Computer Generation of Statistical Distributions
     * by Richard Saucier
     * ARL-TR-2168 March 2000
     * 5.1.8 page 20
     */
    public static double sample(double a, double b, Random random) {
        assert (b > 0.0);
        return a - b * Math.log(random.nextDouble());
    }

    public static double pdf(double a, double b, double x) {
        return (x >= a) ? (1.0 / b) * Math.exp(-(x - a) / b) : 0.0;
    }

    public static double logPdf(double a, double b, double x) {
        return (x >= a) ? (-(x - a) / b) - Math.log(b) : 0.0;
    }

    public static Diff dPdf(double a, double b, double x) {
        double exponent = Math.exp((a - x) / b);
        double bSquared = b * b;

        double dPda = exponent / bSquared;
        double dPdb = -(exponent * (a + b - x)) / (Math.pow(b, 3));
        double dPdx = -dPda;
        return new Diff(dPda, dPdb, dPdx);
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
