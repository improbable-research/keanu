package io.improbable.keanu.distributions.continuous;

import java.util.Random;

import static org.apache.commons.math3.special.Gamma.digamma;
import static org.apache.commons.math3.special.Gamma.gamma;

public class InverseGamma {

    public static double sample(double a, double b, Random random) {
        assert (a > 0.0 && b > 0.0);
        return 1.0 / Gamma.sample(0.0, 1.0 / b, a, random);
    }

    public static double pdf(double a, double b, double x) {
        double numerator = Math.pow(b, a) * Math.pow(x, -a - 1) * Math.exp(-b / x);
        return numerator / gamma(a);
    }

    public static Diff dPdf(double a, double b, double x) {
        double bToThePowerOfA = Math.pow(b, a);
        double eToTheMinusBOverX = Math.exp(-b / x);
        double gammaA = gamma(a);

        double dPda = bToThePowerOfA * Math.pow(x, -a - 1) * eToTheMinusBOverX * (-digamma(a) + Math.log(b) - Math.log(x));
        double dPdb = (Math.pow(b, a - 1) * Math.pow(x, -a - 2) * eToTheMinusBOverX * (a * x - b)) / gammaA;
        double dPdx = (bToThePowerOfA * Math.pow(x, -a - 3) * eToTheMinusBOverX * (b - (a + 1) * x)) / gammaA;

        return new Diff(dPda, dPdb, dPdx);
    }

    public static double logPdf(double a, double b, double x) {
        return a * Math.log(b) + (-a - 1) * Math.log(x) - Math.log(gamma(a)) - (b / x);
    }

    public static Diff dlnPdf(double a, double b, double x) {
        double dPda = -digamma(a) + Math.log(b) - Math.log(x);
        double dPdb = (a / b) - (1 / x);
        double dPdx = (b - (a + 1) * x) / Math.pow(x, 2);
        return new Diff(dPda, dPdb, dPdx);
    }

    public static class Diff {

        public double dPda;
        public double dPdb;
        public double dPdx;

        public Diff(double dPda, double dPdb, double dPdx) {
            this.dPda = dPda;
            this.dPdb = dPdb;
            this.dPdx = dPdx;
        }

    }

}
