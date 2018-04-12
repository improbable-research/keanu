package io.improbable.keanu.distributions.continuous;

public class Laplace {

    public static double sample(double mu, double beta) {
        return 0;
    }

    public static double pdf(double mu, double beta, double x) {
        return 0;
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
