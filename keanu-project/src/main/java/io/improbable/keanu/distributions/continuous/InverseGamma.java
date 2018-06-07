package io.improbable.keanu.distributions.continuous;

import io.improbable.keanu.vertices.dbl.KeanuRandom;

import static org.apache.commons.math3.special.Gamma.digamma;
import static org.apache.commons.math3.special.Gamma.gamma;

public class InverseGamma {

    private InverseGamma() {
    }

    public static double sample(double a, double b, KeanuRandom random) {
        if (a <= 0.0 || b <= 0.0) {
            throw new IllegalArgumentException("Invalid value for a or b. a: " + a + ". b: " + b);
        }
        return 1.0 / Gamma.sample(0.0, 1.0 / b, a, random);
    }

    public static double pdf(double a, double b, double x) {
        double numerator = Math.pow(b, a) * Math.pow(x, -a - 1) * Math.exp(-b / x);
        return numerator / gamma(a);
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
