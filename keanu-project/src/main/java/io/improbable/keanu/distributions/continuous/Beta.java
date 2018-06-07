package io.improbable.keanu.distributions.continuous;


import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static java.lang.Math.pow;
import static org.apache.commons.math3.special.Gamma.*;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.2 page 14
 */

public class Beta {

    private Beta() {
    }

    /**
     * @param alpha  location
     * @param beta   shape
     * @param xMin   minimum x
     * @param xMax   maximum x
     * @param random source of randomness
     * @return a random number from the Beta distribution
     */
    public static double sample(double alpha, double beta, double xMin, double xMax, KeanuRandom random) {
        double y1 = Gamma.sample(0.0, 1.0, alpha, random);
        double y2 = Gamma.sample(0.0, 1.0, beta, random);

        if (alpha < beta) {
            return xMax - (xMax - xMin) * y2 / (y1 + y2);
        } else {
            return xMin + (xMax - xMin) * y1 / (y1 + y2);
        }
    }

    public static double pdf(double alpha, double beta, double x) {
        double denominator = gamma(alpha) * gamma(beta) / gamma(alpha + beta);
        return pow(x, alpha - 1) * pow(1 - x, beta - 1) / denominator;
    }

    public static double logPdf(double alpha, double beta, double x) {
        double betaFunction = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
        return (alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - betaFunction;
    }

    public static Diff dlnPdf(double alpha, double beta, double x) {
        double dPdx = ((alpha - 1) / x) - ((beta - 1) / (1 - x));
        double dPda = digamma(alpha + beta) - digamma(alpha) + Math.log(x);
        double dPdb = digamma(alpha + beta) - digamma(beta) + Math.log(1 - x);

        return new Diff(dPda, dPdb, dPdx);
    }

    public static class Diff {
        public final double dPdalpha;
        public final double dPdbeta;
        public final double dPdx;

        public Diff(double dPdalpha, double dPdbeta, double dPdx) {
            this.dPdalpha = dPdalpha;
            this.dPdbeta = dPdbeta;
            this.dPdx = dPdx;
        }
    }

}
