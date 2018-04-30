package io.improbable.keanu.distributions.continuous;

import java.util.Random;

public class Uniform {

    private Uniform() {
    }

    /**
     * Computer Generation of Statistical Distributions
     * by Richard Saucier
     * ARL-TR-2168 March 2000
     * 5.1.8 page 48
     */

    public static double sample(double xMin, double xMax, Random random) {
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
