package io.improbable.keanu.distributions.continuous;

import java.util.Random;

public class Laplace {

    public static double sample(double mu, double beta, Random random) {
        assert (beta > 0.);
        if (random.nextDouble() > 0.5) {
            return mu + beta * Math.log(random.nextDouble());
        }
        else {
            return mu - beta * Math.log(random.nextDouble());
        }
    }

    public static double pdf(double mu, double beta, double x) {
        return 1 / (2 * beta) * Math.exp(- Math.abs(x - mu) / beta);
    }

    public static Diff dPdf(double mu, double beta, double x) {
        return null;
    }

    public static double logPdf(double mu, double beta, double x) {
        return 0;
    }

    public static Diff dlnPdf(double mu, double beta, double x) {
        return null;
    }

    public static class Diff {
        public final double dPdmu;
        public final double dPdbeta;
        public final double dPdx;

        public Diff(double dPdmu, double dPdbeta, double dPdx) {
            this.dPdmu = dPdmu;
            this.dPdbeta = dPdbeta;
            this.dPdx = dPdx;
        }
    }
}
