package io.improbable.keanu.vertices.dbltensor;

import org.nd4j.linalg.api.rng.Random;

import static java.lang.Math.*;
import static java.lang.Math.exp;
import static java.lang.Math.pow;

class KeanuRandomSampling {

    private KeanuRandomSampling() {
    }

    public static double gammaSample(double a, double theta, double k, Random random) {
        if (theta <= 0. || k <= 0.) {
            throw new IllegalArgumentException("Invalid value for theta or k. Theta: " + theta + ". k: " + k);
        }
        final double M_E = 0.577215664901532860606512090082;
        final double A = 1. / sqrt(2. * k - 1.);
        final double B = k - log(4.);
        final double Q = k + 1. / A;
        final double T = 4.5;
        final double D = 1. + log(T);
        final double C = 1. + k / M_E;

        if (k < 1.) {
            return sampleGammaWhileKLessThanOne(C, k, a, theta, random);
        } else if (k == 1.0) return exponentialSample(a, theta, random);
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

    private static double sampleGammaWhileKLessThanOne(double c, double k, double a, double theta, Random random) {
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

    private static double exponentialSample(double a, double b, Random random) {
        if (b <= 0.0) {
            throw new IllegalArgumentException("Invalid value for b");
        }
        return a - b * Math.log(random.nextDouble());
    }
}
