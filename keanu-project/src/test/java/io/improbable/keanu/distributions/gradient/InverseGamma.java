package io.improbable.keanu.distributions.gradient;

import static org.apache.commons.math3.special.Gamma.digamma;
import static org.apache.commons.math3.special.Gamma.gamma;

public class InverseGamma {

    private InverseGamma() {
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
