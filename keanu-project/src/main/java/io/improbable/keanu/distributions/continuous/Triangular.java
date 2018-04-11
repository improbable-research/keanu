package io.improbable.keanu.distributions.continuous;

import java.util.Random;

public class Triangular {

    /**
     * Computer Generation of Statistical Distributions
     * by Richard Saucier
     * ARL-TR-2168 March 2000
     * 5.1.24 page 37
     */

    public static double sample(double xMin, double xMax, double c, Random random) {
        assert (xMin <= xMax && xMin <= c && c <= xMax);

        double p = random.nextDouble();
        double q = 1.0 - p;

        if (p <= (c - xMin) / (xMax - xMin))
            return xMin + Math.sqrt((xMax - xMin) * (c - xMin) * p);
        else
            return xMax - Math.sqrt((xMax - xMin) * (xMax - c) * q);
    }

    public static double pdf(double xMin, double xMax, double c, double x) {
        double range = xMax - xMin;
        if (x >= xMin && x < c) {
            return (2 / range) * (x - xMin) / (c - xMin);
        } else if (x >= c && x <= xMax) {
            return (2 / range) * (xMax - x) / (xMax - c);
        } else {
            return 0;
        }
    }
}
