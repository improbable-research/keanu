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
        double denominator = 2 * Math.pow(beta, 2) * Math.abs(mu - x);
        double expAbsMuMinusXDivBeta = Math.exp(- (Math.abs(mu - x) / beta));

        double dPdx = ((mu - x) * expAbsMuMinusXDivBeta) / denominator;
        double dPdm = ((x - mu) * expAbsMuMinusXDivBeta) / denominator;
        double dPdb = (expAbsMuMinusXDivBeta * (Math.abs(mu - x) - beta)) / (2 * Math.pow(beta, 3));
        return new Diff(dPdm, dPdb, dPdx);
    }

    public static double logPdf(double mu, double beta, double x) {
        return - (Math.abs(mu - x) + beta * Math.log(2 * beta)) / beta;
    }

    public static Diff dlnPdf(double mu, double beta, double x) {
        double denominator = (beta * Math.abs(mu - x));

        double dPdx = (mu - x) / denominator;
        double dPdm = (x - mu)/ denominator;
        double dPdb = (Math.abs(mu - x) - beta) / Math.pow(beta, 2);
        return new Diff(dPdm, dPdb, dPdx);
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
