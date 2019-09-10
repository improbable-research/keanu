package io.improbable.keanu.algorithms.variational.optimizer.gradient.linesearch;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.FitnessAndGradientFlat;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.FitnessFunctionGradientFlatAdapter;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Value;

public class HagerZhang {

    private final double thresholdUseApproximateWolfeCondition;
    private final double stepSizeShrink;
    private final double rho;
    private final double theta;
    private final double gamma;
    private final double delta;
    private final double sigma;
    private final int maxEvaluations;

    private HagerZhang(double thresholdUseApproximateWolfeCondition,
                       double stepSizeShrink, double rho, double theta,
                       double gamma, double delta, double sigma, int maxEvaluations) {
        this.thresholdUseApproximateWolfeCondition = thresholdUseApproximateWolfeCondition;
        this.stepSizeShrink = stepSizeShrink;
        this.rho = rho;
        this.theta = theta;
        this.gamma = gamma;
        this.delta = delta;
        this.sigma = sigma;
        this.maxEvaluations = maxEvaluations;
    }

    public static HagerZhangBuilder builder() {
        return new HagerZhangBuilder();
    }

    @AllArgsConstructor
    @Data
    public static class Results {
        boolean success;
        double alpha;
    }

    @RequiredArgsConstructor
    private static class Phi {

        private final FitnessFunctionGradientFlatAdapter gradientAdapter;

        private final DoubleTensor x;
        private final DoubleTensor searchDir;

        @Getter
        private int evalCount = 0;

        private EvalResult eval(double alpha) {

            final double[] position = x.plus(searchDir.times(alpha)).asFlatDoubleArray();

            final FitnessAndGradientFlat fitnessAndGradient = gradientAdapter.fitnessAndGradient(position);

            final double df = DoubleTensor.create(fitnessAndGradient.getGradient()).times(searchDir).sum().scalar();
            final double f = fitnessAndGradient.getFitness();

            evalCount++;
            return new EvalResult(-f, -df, alpha);
        }
    }

    public Results lineSearch(DoubleTensor x,
                              DoubleTensor searchDir,
                              FitnessFunctionGradientFlatAdapter objFuncGradient,
                              double initialAlpha) {

        Phi phi = new Phi(objFuncGradient, x, searchDir);

        EvalResult phi0 = phi.eval(0);

        double fLimit = phi0.f + thresholdUseApproximateWolfeCondition * Math.abs(phi0.f);

        EvalResult cInitial = phi.eval(initialAlpha);

        boolean validInputs = phi0.isValid() && phi0.df < 0 && Double.isFinite(cInitial.x) && cInitial.x > 0;
        if (!validInputs) {
            return new Results(false, initialAlpha);
        }

        cInitial = fixStepSize(phi, phi0.f, stepSizeShrink, cInitial);

        Interval currentInterval = bracket(phi, maxEvaluations, theta, phi0, rho, fLimit, cInitial);

        if (veryClose(currentInterval.a.x, currentInterval.b.x)) {
            return new Results(true, currentInterval.a.x);
        }

        while (phi.getEvalCount() < maxEvaluations && !currentInterval.isConverged() && !currentInterval.isFailed()) {
            Interval nextInterval = secant2(phi, maxEvaluations, fLimit, theta, delta, sigma, phi0, currentInterval.a, currentInterval.b);

            boolean shouldCheckShrinkage = !nextInterval.isConverged() && !nextInterval.isFailed();

            if (shouldCheckShrinkage) {

                boolean insufficientShrinkage = nextInterval.b.x - nextInterval.a.x > gamma * (currentInterval.b.x - currentInterval.a.x);

                if (insufficientShrinkage) {

                    final double cPosition = (nextInterval.a.x + nextInterval.b.x) / 2;
                    EvalResult c = phi.eval(cPosition);

                    if (c.isValid()) {
                        nextInterval = update(phi, maxEvaluations, fLimit, theta, nextInterval.a, nextInterval.b, c);

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

        return new Results(!currentInterval.isFailed(), currentInterval.a.x);
    }

    private EvalResult fixStepSize(Phi phi, double f, double stepSizeShrink, EvalResult cValue) {

        if (!cValue.isValid()) {
            double eps = Math.ulp(f);
            int iMax = (int) Math.ceil(-Math.log(eps) / Math.log(2));

            for (int i = 0; i < iMax && !cValue.isValid(); i++) {
                double nextC = stepSizeShrink * cValue.x;
                cValue = phi.eval(nextC);
            }
        }

        return cValue;
    }

    @Value
    @AllArgsConstructor
    public static class EvalResult {
        private final double f;
        private final double df;
        private final double x;

        public boolean isValid() {
            return x >= 0 && Double.isFinite(f) && Double.isFinite(df);
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
        SUCCESS, EXCEEDED_MAX_EVAL, BISECT_FAILED, INSUFFICIENT_SHRINKAGE_FAILED
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
            if (!d.isValid() || d.x == aHat.x || d.x == bHat.x) {
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

        return new Interval(aHat, bHat, Status.EXCEEDED_MAX_EVAL, false);
    }

    private double secant(EvalResult a, EvalResult b) {
        return (a.x * b.df - b.x * a.df) / (b.df - a.df);
    }

    private Interval secant2(Phi phi, int maxEval,
                             double fLimit, double theta, double delta, double sigma, EvalResult phi0,
                             EvalResult a, EvalResult b) {

        double cPosition = secant(a, b);

        if (cPosition > 0) {

            EvalResult c = phi.eval(cPosition);

            if (c.isValid() && satisfiesWolfeConditions(delta, fLimit, sigma, c, phi0)) {
                return new Interval(c, c, Status.SUCCESS, true);
            }

            Interval update = update(phi, maxEval, fLimit, theta, a, b, c);

            if (!update.isFailed() && c.x == update.b.x || c.x == update.a.x) {

                double cHatPosition;
                if (c.x == update.b.x) {
                    cHatPosition = secant(b, update.b);
                } else {
                    cHatPosition = secant(a, update.a);
                }

                if (cHatPosition > 0) {

                    EvalResult cHat = phi.eval(cHatPosition);

                    if (cHat.isValid() && satisfiesWolfeConditions(delta, fLimit, sigma, cHat, phi0)) {
                        return new Interval(cHat, cHat, Status.SUCCESS, true);
                    }

                    return update(phi, maxEval, fLimit, theta, update.a, update.b, cHat);
                }
            }

            return update;

        } else {
            return new Interval(a, b, Status.SUCCESS, false);
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

            if (nextC.isValid()) {
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
        boolean exactWolfeSufficientDecreased = (delta * zero.df >= (c.f - zero.f) / c.x);
        boolean wolfeCurvature = c.df >= sigma * zero.df;
        boolean exactWolfe = exactWolfeSufficientDecreased & wolfeCurvature;
        boolean approxWolfeApplies = c.f <= fLimit;
        boolean approxWolfeSufficientDec = ((2 * delta - 1) * zero.df >= c.df);
        boolean approxWolfe = approxWolfeApplies && approxWolfeSufficientDec && wolfeCurvature;
        return exactWolfe || approxWolfe;
    }


    public static class HagerZhangBuilder {
        private double thresholdUseApproximateWolfeCondition = 1e-6;
        private double stepSizeShrink = 0.1;
        private double rho = 5.0;
        private double theta = 0.5;
        private double gamma = 0.66;
        private double delta = 0.1;
        private double sigma = 0.9;
        private int maxEvaluations = 1000;

        HagerZhangBuilder() {
        }

        public HagerZhangBuilder thresholdUseApproximateWolfeCondition(double thresholdUseApproximateWolfeCondition) {
            this.thresholdUseApproximateWolfeCondition = thresholdUseApproximateWolfeCondition;
            return this;
        }

        public HagerZhangBuilder stepSizeShrink(double stepSizeShrink) {
            this.stepSizeShrink = stepSizeShrink;
            return this;
        }

        public HagerZhangBuilder rho(double rho) {
            this.rho = rho;
            return this;
        }

        public HagerZhangBuilder theta(double theta) {
            this.theta = theta;
            return this;
        }

        public HagerZhangBuilder gamma(double gamma) {
            this.gamma = gamma;
            return this;
        }

        public HagerZhangBuilder delta(double delta) {
            this.delta = delta;
            return this;
        }

        public HagerZhangBuilder sigma(double sigma) {
            this.sigma = sigma;
            return this;
        }

        public HagerZhangBuilder maxEvaluations(int maxEvaluations) {
            this.maxEvaluations = maxEvaluations;
            return this;
        }

        public HagerZhang build() {
            return new HagerZhang(
                thresholdUseApproximateWolfeCondition, stepSizeShrink, rho, theta,
                gamma, delta, sigma, maxEvaluations);
        }
    }
}
