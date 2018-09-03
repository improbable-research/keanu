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

    public static Diff dlnPdf(double theta, double k, double x) {
        double dPdx = (k - 1) / x - (1 / theta);
        double dPdtheta = -(theta * k - x) / Math.pow(theta, 2);
        double dPdk = Math.log(x) - Math.log(theta) - digamma(k);
        return new Diff(dPdtheta, dPdk, dPdx);
    }

    public static class Diff {
        public final double dPdtheta;
        public final double dPdk;
        public final double dPdx;

        public Diff(double dPdtheta, double dPdk, double dPdx) {
            this.dPdtheta = dPdtheta;
            this.dPdk = dPdk;
            this.dPdx = dPdx;
        }
    }
}
