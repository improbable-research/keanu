package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static java.lang.Math.*;
import static org.apache.commons.math3.special.Gamma.digamma;
import static org.apache.commons.math3.special.Gamma.gamma;


/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 33
 */

public class Gamma {

    private Gamma() {
    }

    private static final double M_E = 0.577215664901532860606512090082;

    /**
     * @param a      location
     * @param theta  scale
     * @param k      shape
     * @param random source of randomness
     * @return a random number from the Gamma distribution
     */
    public static double sample(double a, double theta, double k, KeanuRandom random) {
        if (theta <= 0. || k <= 0.) {
            throw new IllegalArgumentException("Invalid value for theta or k. Theta: " + theta + ". k: " + k);
        }
        final double A = 1. / sqrt(2. * k - 1.);
        final double B = k - log(4.);
        final double Q = k + 1. / A;
        final double T = 4.5;
        final double D = 1. + log(T);
        final double C = 1. + k / M_E;

        if (k < 1.) {
            return sampleWhileKLessThanOne(C, k, a, theta, random);
        } else if (k == 1.0) return Exponential.sample(a, theta, random);
        else {
            while (true) {
                double p1 = random.nextDouble();
                double p2 = random.nextDouble();
                double v = A * log(p1 / (1. - p1));
                double y = k * exp(v);
                double z = p1 * p1 * p2;
                double w = B + Q * v - y;
                if (w + D - T * z >= 0. || w >= log(z)) return a + theta * y;
            }
        }
    }

    public static double pdf(double a, double theta, double k, double x) {
        return (pow(theta, -k) * pow(x - a, k - 1) * exp(-(x - a) / theta)) / (gamma(k));
    }

    public static double logPdf(double a, double theta, double k, double x) {
        return (a - x) / theta - (k * Math.log(theta)) + Math.log((Math.pow(x - a, k - 1)) / gamma(k));
    }

    public static Diff dPdf(double a, double theta, double k, double x) {
        double powBminusCminus1 = pow(theta, -k - 1);
        double expAminusXoverB = exp((a - x) / theta);
        double powXminusAToKminus2 = pow(x - a, k - 2);
        double gammaC = gamma(k);
        double commonToDaAndDb = powBminusCminus1 * expAminusXoverB * powXminusAToKminus2;

        double dPdx = (commonToDaAndDb * (a + (theta * (k - 1)) - x)) / gammaC;
        double dPda = (commonToDaAndDb * (theta * (-k) + theta + x - a)) / gammaC;
        double dPdtheta = -(pow(theta, -k - 2) * expAminusXoverB * pow(x - a, k - 1) * (a + (theta * k) - x)) / gammaC;
        double dPdk = -(pow(theta, -k) * expAminusXoverB * pow(x - a, k - 1) * (-log(x - a) + log(theta) + digamma(k))) / gammaC;

        return new Diff(dPda, dPdtheta, dPdk, dPdx);
    }

    public static Diff dlnPdf(double a, double theta, double k, double x) {
        double dPdx = (k - 1) / (x - a) - (1 / theta);
        double dPda = (k - 1) / (a - x) + (1 / theta);
        double dPdtheta = -((a + (theta * k) - x) / Math.pow(theta, 2));
        double dPdk = Math.log(x - a) - Math.log(theta) - digamma(k);
        return new Diff(dPda, dPdtheta, dPdk, dPdx);
    }

    private static double sampleWhileKLessThanOne(double c, double k, double a, double theta, KeanuRandom random) {
        while (true) {
            double p = c * random.nextDouble();
            if (p > 1.) {
                double y = -log((c - p) / k);
                if (random.nextDouble() <= pow(y, k - 1.)) return a + theta * y;
            } else {
                double y = pow(p, 1. / k);
                if (random.nextDouble() <= exp(-y)) return a + theta * y;
            }
        }
    }

    public static class Diff {
        public final double dPda;
        public final double dPdtheta;
        public final double dPdk;
        public final double dPdx;

        public Diff(double dPda, double dPdtheta, double dPdk, double dPdx) {
            this.dPda = dPda;
            this.dPdtheta = dPdtheta;
            this.dPdk = dPdk;
            this.dPdx = dPdx;
        }
    }
}
