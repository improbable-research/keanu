package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.Random;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.14 page 27
 */
public class Logistic {

    private Logistic() {
    }

    /**
     * @param a      location parameter (any real number)
     * @param b      scale parameter (b grater than 0)
     * @param random source or randomness
     * @return a sample from the distribution
     */
    public static double sample(double a, double b, KeanuRandom random) {
        if (b <= 0.0) {
            throw new IllegalArgumentException("Invalid value for beta: " + b);
        }
        return a - b * Math.log(1. / random.nextDouble() - 1.);
    }

    /**
     * @param a location parameter (any real number)
     * @param b scale parameter (b greater than 0)
     * @param x at value
     * @return the density at x
     */
    public static double pdf(double a, double b, double x) {
        double exponential = Math.exp((x - a) / b);
        double denominator = (b * (Math.pow(1 + exponential, 2)));
        return exponential / denominator;
    }

    public static double logPdf(double a, double b, double x) {
        double xMinusAOverB = (x - a) / b;
        return Math.log(1 / b) + xMinusAOverB - (2 * Math.log(1 + Math.exp(xMinusAOverB)));
    }

    /**
     * @param a location parameter (any real number)
     * @param b scale parameter (b greater than 0)
     * @param x at value
     * @return the partial derivatives at x
     */
    public static Diff dPdf(double a, double b, double x) {
        double expAOverB = Math.exp(a / b);
        double expXOverB = Math.exp(x / b);
        double expAPlus2XOverB = Math.exp((a + 2 * x) / b);
        double exp2APlusXOverB = Math.exp((2 * a + x) / b);
        double bSquared = Math.pow(b, 2);

        double dPda = (expAPlus2XOverB - exp2APlusXOverB) / (bSquared * Math.pow(expXOverB + expAOverB, 3));
        double dPdx = (exp2APlusXOverB - expAPlus2XOverB) / (bSquared * Math.pow(expAOverB + expXOverB, 3));
        double dPdb = -(Math.exp((a / b) + (x / b)) * ((a * expXOverB) + (x * expAOverB) - (a * expAOverB) + (b * expAOverB) + (b * expXOverB) - (x * expXOverB))) /
            (Math.pow(b, 3) * Math.pow(expAOverB + expXOverB, 3));

        return new Diff(dPda, dPdb, dPdx);
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
