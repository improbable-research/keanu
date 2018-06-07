package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 48
 */
public class Uniform {

    private Uniform() {
    }

    /**
     * @param xMin   minimum x value
     * @param xMax   maximum x value
     * @param random source of randomness
     * @return a random number from the Uniform distribution
     */
    public static double sample(double xMin, double xMax, KeanuRandom random) {
        return random.nextDouble() * (xMax - xMin) + xMin;
    }

    public static double pdf(double xMin, double xMax, double x) {
        if (x >= xMin && x < xMax) {
            return 1. / (xMax - xMin);
        } else {
            return 0.;
        }
    }
}
