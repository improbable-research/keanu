package io.improbable.keanu.algorithms.variational.optimizer.gradient.linesearch;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.ApacheFitnessFunctionAdapter;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ApacheFitnessFunctionGradientAdapter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

public class MoreThuente {

    private MoreThuente(double xTolerance, double fitnessTolerance, double gradientTolerance, double stepMin, double stepMax, double xtrapf, int maxFitnessEvaluations) {
        this.xTolerance = xTolerance;
        this.fitnessTolerance = fitnessTolerance;
        this.gradientTolerance = gradientTolerance;
        this.stepMin = stepMin;
        this.stepMax = stepMax;
        this.xtrapf = xtrapf;
        this.maxFitnessEvaluations = maxFitnessEvaluations;
    }

    public static MoreThuente withDefaults() {
        return builder().build();
    }

    public static MoreThuenteBuilder builder() {
        return new MoreThuenteBuilder();
    }

    @AllArgsConstructor
    @Data
    public static class Results {
        boolean success;
        double alpha;
    }

    private final double xTolerance;
    private final double fitnessTolerance;
    private final double gradientTolerance;
    private final double stepMin;
    private final double stepMax;
    private final double xtrapf;
    private final int maxFitnessEvaluations;

    public Results lineSearch(DoubleTensor x,
                              DoubleTensor searchDir,
                              ApacheFitnessFunctionAdapter objFunc,
                              ApacheFitnessFunctionGradientAdapter objFuncGradient,
                              double alpha_init) {

        double fitness = objFunc.value(x.asFlatDoubleArray()) * -1;

        DoubleTensor gradient = DoubleTensor.create(objFuncGradient.value(x.asFlatDoubleArray())).unaryMinus();

        return cvsrch(objFunc, objFuncGradient, x, fitness, gradient, alpha_init, searchDir);
    }

    private Results cvsrch(ApacheFitnessFunctionAdapter objFunc,
                           ApacheFitnessFunctionGradientAdapter objFuncGradient,
                           DoubleTensor x,
                           double fitness,
                           DoubleTensor gradient,
                           final double stp,
                           DoubleTensor searchDirection) {


        final double finit = fitness;
        final DoubleTensor initialX = x.duplicate();

        double dginit = dot(gradient, searchDirection).scalar();
        if (dginit >= 0.0) {
            // no descent direction
            // TODO: handle this case
            return new Results(false, stp);
        }

        final double dgtest = fitnessTolerance * dginit;

        int info = 0;
        int nfev = 0;
        boolean stage1 = true;
        double width = stepMax - stepMin;
        double width1 = 2 * width;

        final CStep cStep = new CStep();
        cStep.stp = stp;
        cStep.info = 1;
        cStep.brackt = false;
        cStep.stx = 0.0;
        cStep.fx = finit;
        cStep.dx = dginit;
        cStep.sty = 0.0;
        cStep.fy = finit;
        cStep.dy = dginit;

        double stmin;
        double stmax;

        while (true) {

            // make sure we stay in the interval when setting min/max-step-width
            if (cStep.brackt) {
                stmin = Math.min(cStep.stx, cStep.sty);
                stmax = Math.max(cStep.stx, cStep.sty);
            } else {
                stmin = cStep.stx;
                stmax = cStep.stp + xtrapf * (cStep.stp - cStep.stx);
            }

            // Oops, let us return the last reliable values
            if ((cStep.brackt && ((cStep.stp <= stmin) || (cStep.stp >= stmax)))
                || (nfev >= maxFitnessEvaluations - 1) || (cStep.info == 0)
                || (cStep.brackt && ((stmax - stmin) <= (xTolerance * stmax)))) {
                cStep.stp = cStep.stx;
            }

            // Force the step to be within the bounds stpmax and stpmin.
            cStep.stp = Math.max(cStep.stp, stepMin);
            cStep.stp = Math.min(cStep.stp, stepMax);

            // test new point
            x = initialX.plus(searchDirection.times(cStep.stp));
            fitness = objFunc.value(x.asFlatDoubleArray()) * -1;
            gradient = DoubleTensor.create(objFuncGradient.value(x.asFlatDoubleArray())).unaryMinus();

            if (fitness == Double.POSITIVE_INFINITY) {
                fitness = 1e20;
            }

            gradient = gradient.replaceNaN(0.0);

            nfev++;

            final double dg = dot(gradient, searchDirection).scalar();
            final double ftest1 = finit + cStep.stp * dgtest;

            // all possible convergence tests
            if ((cStep.brackt & ((cStep.stp <= stmin) | (cStep.stp >= stmax))) | (cStep.info == 0)) {
                info = 6;
            }

            if ((cStep.stp == stepMax) & (fitness <= ftest1) & (dg <= dgtest)) {
                info = 5;
            }

            if ((cStep.stp == stepMin) & ((fitness > ftest1) | (dg >= dgtest))) {
                info = 4;
            }

            if (nfev >= maxFitnessEvaluations) {
                info = 3;
            }

            if (cStep.brackt & (stmax - stmin <= xTolerance * stmax)) {
                info = 2;
            }

            if ((fitness <= ftest1) & (Math.abs(dg) <= gradientTolerance * (-dginit))) {
                info = 1;
            }

            // terminate when convergence reached
            if (info != 0) {
                return new Results(true, cStep.stp);
            }

            if (stage1 & (fitness <= ftest1) & (dg >= Math.min(fitnessTolerance, gradientTolerance) * dginit)) {
                stage1 = false;
            }

            if (stage1 & (fitness <= cStep.fx) & (fitness > ftest1)) {
                final double fm = fitness - cStep.stp * dgtest;
                final double fxm = cStep.fx - cStep.stx * dgtest;
                final double fym = cStep.fy - cStep.sty * dgtest;
                final double dgm = dg - dgtest;
                final double dgxm = cStep.dx - dgtest;
                final double dgym = cStep.dy - dgtest;

                final CStep newCstep = new CStep(cStep.stx, fxm, dgxm, cStep.sty, fym, dgym, cStep.stp, cStep.brackt, cStep.info);

                cstep(newCstep, fm, dgm, stmin, stmax);

                cStep.fx = newCstep.fx + newCstep.stx * dgtest;
                cStep.fy = newCstep.fy + newCstep.sty * dgtest;
                cStep.dx = newCstep.dx + dgtest;
                cStep.dy = newCstep.dy + dgtest;

                cStep.info = newCstep.info;
                cStep.brackt = newCstep.brackt;
                cStep.stx = newCstep.stx;
                cStep.sty = newCstep.sty;
                cStep.stp = newCstep.stp;

            } else {

                cstep(cStep, fitness, dg, stmin, stmax);
            }

            if (cStep.brackt) {
                if (Math.abs(cStep.sty - cStep.stx) >= 0.66 * width1) {
                    cStep.stp = cStep.stx + 0.5 * (cStep.sty - cStep.stx);
                }
                width1 = width;
                width = Math.abs(cStep.sty - cStep.stx);
            }
        }

    }

    private static void cstep(final CStep cStep, final double fp, final double dp, final double stpmin, final double stpmax) {

        final double stx = cStep.stx;
        final double fx = cStep.fx;
        final double dx = cStep.dx;
        final double sty = cStep.sty;
        final double fy = cStep.fy;
        final double dy = cStep.dy;
        final double stp = cStep.stp;
        final boolean brackt = cStep.brackt;

        cStep.info = 0;
        boolean bound;

        // Check the input parameters for errors.
        if ((brackt & ((stp <= Math.min(stx, sty)) | (stp >= Math.max(stx, sty)))) | (dx * (stp - stx) >= 0.0)
            | (stpmax < stpmin)) {
            return;
        }

        final double sgnd = dp * (dx / Math.abs(dx));

        double stpf = 0;
        double stpc = 0;
        double stpq = 0;

        if (fp > fx) {

            cStep.info = 1;
            bound = true;
            final double theta = 3. * (fx - fp) / (stp - stx) + dx + dp;
            final double s = Math.max(theta, Math.max(dx, dp));
            double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));

            if (stp < stx) {
                gamma = -gamma;
            }

            final double p = (gamma - dx) + theta;
            final double q = ((gamma - dx) + gamma) + dp;
            final double r = p / q;

            stpc = stx + r * (stp - stx);
            stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.) * (stp - stx);

            if (Math.abs(stpc - stx) < Math.abs(stpq - stx)) {
                stpf = stpc;
            } else {
                stpf = stpc + (stpq - stpc) / 2;
            }
            cStep.brackt = true;

        } else if (sgnd < 0.0) {

            cStep.info = 2;
            bound = false;
            final double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
            final double s = Math.max(theta, Math.max(dx, dp));

            double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp > stx) {
                gamma = -gamma;
            }

            final double p = (gamma - dp) + theta;
            final double q = ((gamma - dp) + gamma) + dx;
            final double r = p / q;

            stpc = stp + r * (stx - stp);
            stpq = stp + (dp / (dp - dx)) * (stx - stp);

            if (Math.abs(stpc - stp) > Math.abs(stpq - stp)) {
                stpf = stpc;
            } else {
                stpf = stpq;
            }
            cStep.brackt = true;

        } else if (Math.abs(dp) < Math.abs(dx)) {

            cStep.info = 3;
            bound = true;
            final double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
            final double s = Math.max(theta, Math.max(dx, dp));

            double gamma = s * Math.sqrt(Math.max(0., (theta / s) * (theta / s) - (dx / s) * (dp / s)));
            if (stp > stx) {
                gamma = -gamma;
            }

            final double p = (gamma - dp) + theta;
            final double q = (gamma + (dx - dp)) + gamma;
            final double r = p / q;

            if ((r < 0.0) & (gamma != 0.0)) {
                stpc = stp + r * (stx - stp);
            } else if (stp > stx) {
                stpc = stpmax;
            } else {
                stpc = stpmin;
            }

            stpq = stp + (dp / (dp - dx)) * (stx - stp);

            if (brackt) {
                if (Math.abs(stp - stpc) < Math.abs(stp - stpq)) {
                    stpf = stpc;
                } else {
                    stpf = stpq;
                }
            } else {
                if (Math.abs(stp - stpc) > Math.abs(stp - stpq)) {
                    stpf = stpc;
                } else {
                    stpf = stpq;
                }

            }
        } else {

            cStep.info = 4;
            bound = false;

            if (brackt) {
                final double theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
                final double s = Math.max(theta, Math.max(dy, dp));

                double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
                if (stp > sty) {
                    gamma = -gamma;
                }

                final double p = (gamma - dp) + theta;
                final double q = ((gamma - dp) + gamma) + dy;
                final double r = p / q;
                stpc = stp + r * (sty - stp);
                stpf = stpc;

                if (Double.isNaN(stpf)) {
                    System.out.println();
                }

            } else if (stp > stx)
                stpf = stpmax;
            else {
                stpf = stpmin;
            }

        }

        if (fp > fx) {
            cStep.sty = stp;
            cStep.fy = fp;
            cStep.dy = dp;
        } else {
            if (sgnd < 0.0) {
                cStep.sty = stx;
                cStep.fy = fx;
                cStep.dy = dx;
            }

            cStep.stx = stp;
            cStep.fx = fp;
            cStep.dx = dp;
        }

        stpf = Math.min(stpmax, stpf);
        stpf = Math.max(stpmin, stpf);
        cStep.stp = stpf;

        if (brackt & bound) {
            if (sty > stx) {
                cStep.stp = Math.min(stx + 0.66 * (sty - stx), stp);
            } else {
                cStep.stp = Math.max(stx + 0.66 * (sty - stx), stp);
            }
        }

        if (Double.isNaN(cStep.stp)) {
            System.out.println();
        }
    }

    private static DoubleTensor dot(DoubleTensor left, DoubleTensor right) {
        return left.times(right).sum();
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    private static class CStep {
        double stx, fx, dx, sty, fy, dy, stp;
        boolean brackt;
        int info;
    }

    public static class MoreThuenteBuilder {
        private double xTolerance = 1e-15;
        private double fitnessTolerance = 1e-4;
        private double gradientTolerance = 1e-2;
        private double stepMin = 1e-8;
        private double stepMax = 1e8;
        private double xtrapf = 4;
        private int maxFitnessEvaluations = 20;

        MoreThuenteBuilder() {
        }

        public MoreThuenteBuilder xTolerance(double xTolerance) {
            this.xTolerance = xTolerance;
            return this;
        }

        public MoreThuenteBuilder fitnessTolerance(double fitnessTolerance) {
            this.fitnessTolerance = fitnessTolerance;
            return this;
        }

        public MoreThuenteBuilder gradientTolerance(double gradientTolerance) {
            this.gradientTolerance = gradientTolerance;
            return this;
        }

        public MoreThuenteBuilder stepMin(double stepMin) {
            this.stepMin = stepMin;
            return this;
        }

        public MoreThuenteBuilder stepMax(double stepMax) {
            this.stepMax = stepMax;
            return this;
        }

        public MoreThuenteBuilder xtrapf(double xtrapf) {
            this.xtrapf = xtrapf;
            return this;
        }

        public MoreThuenteBuilder maxFitnessEvaluations(int maxFitnessEvaluations) {
            this.maxFitnessEvaluations = maxFitnessEvaluations;
            return this;
        }

        public MoreThuente build() {
            return new MoreThuente(xTolerance, fitnessTolerance, gradientTolerance, stepMin, stepMax, xtrapf, maxFitnessEvaluations);
        }

        public String toString() {
            return "MoreThuente.MoreThuenteBuilder(xTolerance=" + this.xTolerance + ", fitnessTolerance=" + this.fitnessTolerance + ", gradientTolerance=" + this.gradientTolerance + ", stepMin=" + this.stepMin + ", stepMax=" + this.stepMax + ", xtrapf=" + this.xtrapf + ", maxFitnessEvaluations=" + this.maxFitnessEvaluations + ")";
        }
    }

}
