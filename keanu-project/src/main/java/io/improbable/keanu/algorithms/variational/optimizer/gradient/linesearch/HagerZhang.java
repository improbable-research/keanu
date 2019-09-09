package io.improbable.keanu.algorithms.variational.optimizer.gradient.linesearch;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.ApacheFitnessFunctionAdapter;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.ApacheFitnessFunctionGradientAdapter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

public class HagerZhang {

    private final double thresholdUseApproximateWolfeCondition = 1e-6;

    private final double stepSizeShrink = 0.1;

    private final double rho = 5.0;

    private final double theta = 0.5;

    private final double gamma = 0.66;

    private final double delta = 0.1;

    private final double sigma = 0.9;

    @AllArgsConstructor
    @Data
    public static class Results {
        boolean success;
        double alpha;
    }

    @RequiredArgsConstructor
    private static class Phi {

        private final ApacheFitnessFunctionAdapter fitnessFunction;
        private final ApacheFitnessFunctionGradientAdapter gradientAdapter;

        private final DoubleTensor x;
        private final DoubleTensor searchDir;

        @Getter
        private int evalCount = 0;

        private EvalResult eval(double alpha) {

            final double df = DoubleTensor.create(gradientAdapter.value(x.plus(searchDir.times(alpha)).asFlatDoubleArray()))
                .times(searchDir).sum().scalar();

            final double f = fitnessFunction.value(x.plus(searchDir.times(alpha)).asFlatDoubleArray());

            evalCount++;
            return new EvalResult(-f, -df, alpha);
        }
    }

    public Results lineSearch(DoubleTensor x,
                              DoubleTensor searchDir,
                              ApacheFitnessFunctionAdapter objFunc,
                              ApacheFitnessFunctionGradientAdapter objFuncGradient,
                              double initialAlpha) {

        final int maxEval = 1000;
        Phi phi = new Phi(objFunc, objFuncGradient, x, searchDir);

        EvalResult phi0 = phi.eval(0);

        double fLimit = phi0.f + thresholdUseApproximateWolfeCondition * Math.abs(phi0.f);

        EvalResult cInitial = phi.eval(initialAlpha);

        boolean validInputs = phi0.isFinite() && phi0.df < 0 && Double.isFinite(cInitial.x) && cInitial.x > 0;
        if (!validInputs) {
            return new Results(false, initialAlpha);
        }

        cInitial = fixStepSize(phi, phi0.f, stepSizeShrink, cInitial);

        Interval currentInterval = bracket(phi, maxEval, theta, phi0, rho, fLimit, cInitial);

        if (veryClose(currentInterval.a.x, currentInterval.b.x)) {
            return new Results(true, currentInterval.a.x);
        }

        while (phi.getEvalCount() < maxEval && !currentInterval.isConverged() && !currentInterval.isFailed()) {
            Interval nextInterval = secant2(phi, maxEval, fLimit, theta, delta, sigma, phi0, currentInterval.a, currentInterval.b);

            boolean shouldCheckShrinkage = !nextInterval.isConverged() && !nextInterval.isFailed();

            if (shouldCheckShrinkage) {

                boolean insufficientShrinkage = nextInterval.b.x - nextInterval.a.x > gamma * (currentInterval.b.x - currentInterval.a.x);

                if (insufficientShrinkage) {

                    final double cPosition = (nextInterval.a.x + nextInterval.b.x) / 2;
                    EvalResult c = phi.eval(cPosition);

                    if (c.isFinite()) {
                        nextInterval = update(phi, maxEval, fLimit, theta, nextInterval.a, nextInterval.b, c);

                    } else {
                        nextInterval = new Interval(nextInterval.a, nextInterval.b, Status.INSUFFICIENT_SHRINKAGE_FAILED, false);
                    }

                } else {

                    boolean isFlat = veryClose(currentInterval.a.f, currentInterval.b.f) && veryClose(nextInterval.a.f, nextInterval.b.f);
                    nextInterval = new Interval(nextInterval.a, nextInterval.b, Status.SUCCESS, isFlat);
                }
            }

            if (veryClose(nextInterval.a.x, nextInterval.b.x) && !nextInterval.isConverged()) {
                currentInterval = new Interval(nextInterval.a, nextInterval.b, Status.SUCCESS, true);
            } else {

                currentInterval = nextInterval;
            }
        }

        if (currentInterval.isFailed()) {
            System.out.println("Failed " + currentInterval.status);
        }

        if (currentInterval.a.x <= 0) {
            System.out.println("wtf");
        }

        return new Results(!currentInterval.isFailed(), currentInterval.a.x);
    }

    private EvalResult fixStepSize(Phi phi, double f, double stepSizeShrink, EvalResult cValue) {

        if (!cValue.isFinite()) {
            double eps = Math.ulp(f);
            int iMax = (int) Math.ceil(-Math.log(eps) / Math.log(2));

            for (int i = 0; i < iMax && !cValue.isFinite(); i++) {
                double nextC = stepSizeShrink * cValue.x;
                cValue = phi.eval(nextC);
            }
        }

        return cValue;
    }

    @Data
    @AllArgsConstructor
    public static class EvalResult {
        double f;
        double df;
        double x;

        public boolean isFinite() {
            return Double.isFinite(f) && Double.isFinite(df);
        }
    }

    @Data
    @AllArgsConstructor
    public static class Interval {
        EvalResult a;
        EvalResult b;
        Status status;
        boolean converged;

        public boolean isFailed() {
            return status != Status.SUCCESS;
        }
    }

    private enum Status {
        SUCCESS, EXCEEDED_MAX_EVAL, BISECT_FAILED, SECANT2_FAILED, INSUFFICIENT_SHRINKAGE_FAILED
    }

    private Interval update(Phi phi, int maxEval, double fLimit, double theta,
                            EvalResult a, EvalResult b, EvalResult c) {

        if (c.x > b.x || c.x < a.x) {
            //out of range
            return new Interval(a, b, Status.SUCCESS, false);
        } else if (c.df >= 0) {
            //new right bound
            return new Interval(a, c, Status.SUCCESS, false);
        } else {

            if (c.f <= fLimit) {
                // new left limit
                return new Interval(c, b, Status.SUCCESS, false);
            } else {

                //needs bisect
                return bisect(phi, maxEval, theta, fLimit, a, c);
            }
        }
    }

    /**
     * Corresponds to U3 from Hager Zhang paper
     *
     * @param phi    Fitness and gradient function that is in terms of alpha
     * @param theta  The bisect balance. 0.5 is an actual bisect.
     * @param fLimit The fitness with alpha=0 plus the approx wolfe condition threshold
     * @param aHat   left side of interval
     * @param bHat   right side of interval
     * @return The interval continuously bisected until the right side gradient is positive
     */
    private Interval bisect(Phi phi, int maxEval, double theta, double fLimit,
                            EvalResult aHat, EvalResult bHat) {

        while (phi.getEvalCount() < maxEval) {

            final double dPosition = (1 - theta) * aHat.x + theta * bHat.x;
            final EvalResult d = phi.eval(dPosition);

            //If the midpoint is not finite or cannot be distinguished from the
            //left (aHat) or right (bHat) then bisect has failed.
            if (!d.isFinite() || d.x == aHat.x || d.x == bHat.x) {
                return new Interval(aHat, bHat, Status.BISECT_FAILED, false);
            }

            if (d.df < 0) {
                if (d.f <= fLimit) {
                    aHat = d;
                } else {
                    bHat = d;
                }
            } else {
                return new Interval(aHat, d, Status.SUCCESS, false);
            }

        }

        return new Interval(null, null, Status.EXCEEDED_MAX_EVAL, false);
    }

    private EvalResult secant(Phi phi,
                              EvalResult a, EvalResult b) {
        double cPosition = (a.x * b.df - b.x * a.df) / (b.df - a.df);
        return phi.eval(cPosition);
    }

    private Interval secant2(Phi phi, int maxEval,
                             double fLimit, double theta, double delta, double sigma, EvalResult phi0,
                             EvalResult a, EvalResult b) {

        EvalResult c = secant(phi, a, b);

        if (c.isFinite() && satisfiesWolfeConditions(delta, fLimit, sigma, c, phi0)) {
            return new Interval(c, c, Status.SUCCESS, true);
        }

        Interval update = update(phi, maxEval, fLimit, theta, a, b, c);

        if (c.x == update.b.x || c.x == update.a.x) {

            EvalResult cHat;
            if (c.x == update.b.x) {
                cHat = secant(phi, b, update.b);
            } else {
                cHat = secant(phi, a, update.a);
            }

            if (cHat.isFinite() && satisfiesWolfeConditions(delta, fLimit, sigma, cHat, phi0)) {
                return new Interval(cHat, cHat, Status.SUCCESS, true);
            }

            return update(phi, maxEval, fLimit, theta, update.a, update.b, cHat);
        } else {
            return update;
        }

    }

    private Interval bracket(Phi phi, int maxEval,
                             double theta, EvalResult phi0, double rho,
                             double fLimit, EvalResult c) {

        EvalResult ci = phi0;

        while (phi.getEvalCount() < maxEval) {

            if (c.df >= 0) {
                //bracketed
                return new Interval(ci, c, Status.SUCCESS, false);
            } else if (c.f > fLimit) {
                //needs bisect
                return bisect(phi, maxEval, theta, fLimit, phi0, c);
            }

            EvalResult nextC = phi.eval(c.x * rho);

            if (nextC.isFinite()) {
                ci = c;
                c = nextC;
            } else {
                //stop growing due to reaching non-finite range
                return new Interval(ci, c, Status.SUCCESS, false);
            }
        }

        return new Interval(ci, c, Status.EXCEEDED_MAX_EVAL, false);
    }

    private boolean veryClose(double x, double y) {
        return Math.nextAfter(x, y) >= y;
    }

    private boolean satisfiesWolfeConditions(double delta, double fLimit, double sigma, EvalResult c, EvalResult zero) {

        if (c.x <= 0) {
            return false;
        }

        boolean exactWolfeSufficientDecreased = (delta * zero.df >= (c.f - zero.f) / c.x);
        boolean wolfeCurvature = c.df >= sigma * zero.df;
        boolean exactWolfe = exactWolfeSufficientDecreased & wolfeCurvature;
        boolean approxWolfeApplies = c.f <= fLimit;
        boolean approxWolfeSufficientDec = ((2 * delta - 1) * zero.df >= c.df);
        boolean approxWolfe = approxWolfeApplies && approxWolfeSufficientDec && wolfeCurvature;
        return exactWolfe || approxWolfe;
    }


}
