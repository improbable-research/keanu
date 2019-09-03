package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import static io.improbable.keanu.algorithms.variational.optimizer.gradient.LBFGS.dot;

public class MoreThuente {

    static double linesearch(DoubleTensor x,
                             DoubleTensor searchDir,
                             ApacheFitnessFunctionAdapter objFunc,
                             ApacheFitnessFunctionGradientAdapter objFuncGradient,
                             double alpha_init) {

        double fitness = objFunc.value(x.asFlatDoubleArray()) * -1;

        DoubleTensor g = DoubleTensor.create(objFuncGradient.value(x.asFlatDoubleArray())).unaryMinus();

        return cvsrch(objFunc, objFuncGradient, x, fitness, g, alpha_init, searchDir);
    }

    private static double cvsrch(ApacheFitnessFunctionAdapter objFunc,
                                 ApacheFitnessFunctionGradientAdapter objFuncGradient,
                                 DoubleTensor x,
                                 double f,
                                 DoubleTensor g,
                                 final double stp,
                                 DoubleTensor searchDirection) {

        CStep cStep = new CStep();
        cStep.stp = stp;

        int info = 0;
        cStep.info = 1;
        double xtol = 1e-15;
        double ftol = 1e-4;
        double gtol = 1e-2;
        final double stpmin = 1e-15;
        final double stpmax = 1e15;
        double xtrapf = 4;
        final int maxfev = 20;
        int nfev = 0;

        double dginit = dot(g, searchDirection).scalar();
        if (dginit >= 0.0) {
            // no descent direction
            // TODO: handle this case
            return -1;
        }

        cStep.brackt = false;
        boolean stage1 = true;

        double finit = f;
        double dgtest = ftol * dginit;
        double width = stpmax - stpmin;
        double width1 = 2 * width;
        DoubleTensor wa = x.duplicate();

        cStep.stx = 0.0;
        cStep.fx = finit;
        cStep.dx = dginit;
        cStep.sty = 0.0;
        cStep.fy = finit;
        cStep.dy = dginit;

        double stmin = 0.0, stmax =0.0;

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
                    || (nfev >= maxfev - 1) || (cStep.info == 0)
                    || (cStep.brackt && ((stmax - stmin) <= (xtol * stmax)))) {
                cStep.stp = cStep.stx;
            }

            // Force the step to be within the bounds stpmax and stpmin.
            cStep.stp = Math.max(cStep.stp, stpmin);
            cStep.stp = Math.min(cStep.stp, stpmax);

            // test new point
            x = wa.plus(searchDirection.times(cStep.stp));
            f = objFunc.value(x.asFlatDoubleArray()) * -1;
            g = DoubleTensor.create(objFuncGradient.value(x.asFlatDoubleArray())).unaryMinus();
            nfev++;
            double dg = dot(g, searchDirection).scalar();
            double ftest1 = finit + cStep.stp * dgtest;

            // all possible convergence tests
            if ((cStep.brackt & ((cStep.stp <= stmin) | (cStep.stp >= stmax))) | (cStep.info == 0)) {
                info = 6;
            }

            if ((cStep.stp == stpmax) & (f <= ftest1) & (dg <= dgtest)) {
                info = 5;
            }

            if ((cStep.stp == stpmin) & ((f > ftest1) | (dg >= dgtest))) {
                info = 4;
            }

            if (nfev >= maxfev) {
                info = 3;
            }

            if (cStep.brackt & (stmax - stmin <= xtol * stmax)) {
                info = 2;
            }

            if ((f <= ftest1) & (Math.abs(dg) <= gtol * (-dginit))) {
                info = 1;
            }

            // terminate when convergence reached
            if (info != 0) {
                if(cStep.stp == 0){
                    System.out.println("");
                }
                return cStep.stp;
            }

            if (stage1 & (f <= ftest1) & (dg >= Math.min(ftol, gtol) * dginit)) {
                stage1 = false;
            }

            if (stage1 & (f <= cStep.fx) & (f > ftest1)) {
                double fm = f - cStep.stp * dgtest;
                double fxm = cStep.fx - cStep.stx * dgtest;
                double fym = cStep.fy - cStep.sty * dgtest;
                double dgm = dg - dgtest;
                double dgxm = cStep.dx - dgtest;
                double dgym = cStep.dy - dgtest;

                CStep newCstep = new CStep(cStep.stx, fxm, dgxm, cStep.sty, fym, dgym, cStep.stp, cStep.brackt, cStep.info);

                //cstep( stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin, stmax, infoc);

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
                // this is ugly and some variables should be moved to the class scope
                cstep(cStep, f, dg, stmin, stmax);
//                cstep(cStep.stx, cStep.fx, cStep.dx, cStep.sty, cStep.fy, cStep.dy, cStep.stp, f, dg, cStep.brackt, cStep.stpmin, cStep.stpmax, cStep.info);
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

//    static int cstep(double stx,
//                     double fx,
//                     double dx,
//                     double sty,
//                     double fy,
//                     double dy,
//                     double stp,
//                     double fp,
//                     double dp,
//                     boolean brackt,
//                     double stpmin,
//                     double stpmax,
//                     int info
//    ) {

    static int cstep(CStep cStep, final double fp, final double dp, final double stpmin, final double stpmax) {

        final double stx = cStep.stx;
        final double fx = cStep.fx;
        final double dx = cStep.dx;
        final double sty = cStep.sty;
        final double fy = cStep.fy;
        final double dy = cStep.dy;
        final double stp = cStep.stp;
        final boolean brackt = cStep.brackt;

        cStep.info = 0;
        boolean bound = false;

        // Check the input parameters for errors.
        if ((brackt & ((stp <= Math.min(stx, sty)) | (stp >= Math.max(stx, sty)))) | (dx * (stp - stx) >= 0.0)
            | (stpmax < stpmin)) {
            return -1;
        }

        double sgnd = dp * (dx / Math.abs(dx));
        double stpf = 0;
        double stpc = 0;
        double stpq = 0;

        if (fp > fx) {
            cStep.info = 1;
            bound = true;
            double theta = 3. * (fx - fp) / (stp - stx) + dx + dp;
            double s = Math.max(theta, Math.max(dx, dp));
            double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp < stx)
                gamma = -gamma;
            double p = (gamma - dx) + theta;
            double q = ((gamma - dx) + gamma) + dp;
            double r = p / q;
            stpc = stx + r * (stp - stx);
            stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.) * (stp - stx);
            if (Math.abs(stpc - stx) < Math.abs(stpq - stx)) {
                stpf = stpc;
            }else {
                stpf = stpc + (stpq - stpc) / 2;
            }
            cStep.brackt = true;
        } else if (sgnd < 0.0) {
            cStep.info = 2;
            bound = false;
            double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
            double s = Math.max(theta, Math.max(dx, dp));
            double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
            if (stp > stx) {
                gamma = -gamma;
            }

            double p = (gamma - dp) + theta;
            double q = ((gamma - dp) + gamma) + dx;
            double r = p / q;
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
            double theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
            double s = Math.max(theta, Math.max(dx, dp));
            double gamma = s * Math.sqrt(Math.max(0., (theta / s) * (theta / s) - (dx / s) * (dp / s)));
            if (stp > stx) {
                gamma = -gamma;
            }
            double p = (gamma - dp) + theta;
            double q = (gamma + (dx - dp)) + gamma;
            double r = p / q;
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
                double theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
                double s = Math.max(theta, Math.max(dy, dp));
                double gamma = s * Math.sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
                if (stp > sty) {
                    gamma = -gamma;
                }

                double p = (gamma - dp) + theta;
                double q = ((gamma - dp) + gamma) + dy;
                double r = p / q;
                stpc = stp + r * (sty - stp);
                stpf = stpc;
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

            if(cStep.stx == 0){
                System.out.println();
            }
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

        return 0;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    private static class CStep {
        double stx, fx, dx, sty, fy, dy, stp;
        boolean brackt;
        int info;
    }
}
