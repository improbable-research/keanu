package io.improbable.keanu.distributions.gradient;

import static org.apache.commons.math3.special.Gamma.digamma;


/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 33
 */

public class Gamma {

    private Gamma() {
    }

    public static Diff dlnPdf(double a, double theta, double k, double x) {
        double dPdx = (k - 1) / (x - a) - (1 / theta);
        double dPda = (k - 1) / (a - x) + (1 / theta);
        double dPdtheta = -((a + (theta * k) - x) / Math.pow(theta, 2));
        double dPdk = Math.log(x - a) - Math.log(theta) - digamma(k);
        return new Diff(dPda, dPdtheta, dPdk, dPdx);
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
