package io.improbable.keanu.distributions.gradient;


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
