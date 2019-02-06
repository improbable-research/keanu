package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.SaveStatistics;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


/**
 * Used by NUTS as an epsilon for the leap frog
 */
class Stepsize implements SaveStatistics {

    private static final double t0 = 10;
    private static final double gamma = 0.05;
    private static final double kappa = 0.75;
    private static final double STARTING_STEPSIZE = 1;

    private final double mu;
    private final double adaptCount;
    private final double sigma;

    private double stepsize;
    private double hBar;
    private double acceptRate;
    private double logStepSizeFrozen;
    private double logStepSize;

    /**
     * @param stepsize   the step size
     * @param sigma      the target acceptance probability (lower target equates to a higher step size when tuning)
     * @param adaptCount the number of samples to adapt for
     */
    Stepsize(double stepsize, double sigma, int adaptCount) {
        this.sigma = sigma;
        this.stepsize = stepsize;
        this.hBar = 0;
        this.logStepSize = Math.log(stepsize);
        this.logStepSizeFrozen = logStepSize;//Math.log(1);
        this.adaptCount = adaptCount;
        this.mu = Math.log(10 * stepsize);
    }

    public static double findStartingStepSizeSimple(double stepScale, List<? extends Variable<DoubleTensor, ?>> variables) {
        long N = variables.stream()
            .mapToLong(v -> v.getValue().getLength())
            .sum();

        return stepScale / Math.pow(N, 0.25);
    }

//    /**
//     * Taken from algorithm 4 in https://arxiv.org/pdf/1111.4246.pdf.
//     *
//     * @param position                       the starting position
//     * @param gradient                       the gradient at the starting position
//     * @param variables                      the variables
//     * @param probabilisticModelWithGradient the probabilistic model with gradient
//     * @param initialLogOfMasterP            the initial master log prob
//     * @param random                         the source of randomness
//     * @return a starting step size
//     */
//    public static double findStartingStepSize(Map<VariableReference, DoubleTensor> position,
//                                              Map<? extends VariableReference, DoubleTensor> gradient,
//                                              List<? extends Variable<DoubleTensor, ?>> variables,
//                                              ProbabilisticModelWithGradient probabilisticModelWithGradient,
//                                              double initialLogOfMasterP,
//                                              KeanuRandom random) {
//        double stepsize = STARTING_STEPSIZE;
//        Map<VariableReference, DoubleTensor> momentums = variables.stream()
//            .collect(Collectors.toMap(
//                Variable::getReference,
//                v -> random.nextGaussian(v.getShape())
//            ));
//
//        Leapfrog leapfrog = new Leapfrog(position, momentums, gradient);
//        double pThetaR = initialLogOfMasterP - leapfrog.halfDotProductMomentum();
//
//        Leapfrog delta = leapfrog.step(variables, probabilisticModelWithGradient, STARTING_STEPSIZE);
//
//        double probAfterLeapfrog = probabilisticModelWithGradient.logProb();
//        double pThetaRAfterLeapFrog = probAfterLeapfrog - delta.halfDotProductMomentum();
//
//        double logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
//        double scalingFactor = logLikelihoodRatio > Math.log(0.5) ? 1 : -1;
//
//        while (scalingFactor * logLikelihoodRatio > -scalingFactor * Math.log(2)) {
//            stepsize = stepsize * Math.pow(2, scalingFactor);
//
//            delta = leapfrog.step(variables, probabilisticModelWithGradient, stepsize);
//            probAfterLeapfrog = probabilisticModelWithGradient.logProb();
//            pThetaRAfterLeapFrog = probAfterLeapfrog - delta.halfDotProductMomentum();
//
//            logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
//        }
//
//        return stepsize;
//    }

    /**
     * Adapts the step size based on the state of the tree and network after computing a sample
     *
     * @param tree      the balanced binary tree
     * @param sampleNum the number of samples that have been taken
     * @return a new step size
     */
    public double adaptStepSize(Tree tree, int sampleNum) {

        final double logStepSizeAtSample;
        if (sampleNum < adaptCount) {
            logStepSizeAtSample = updateLogStepSize(tree, sampleNum);
        } else {
            logStepSizeAtSample = logStepSizeFrozen;
        }

        stepsize = Math.exp(logStepSizeAtSample);
        return stepsize;
    }

    private double updateLogStepSize(Tree tree, int m) {

        final double alpha = tree.getDeltaLikelihoodOfLeapfrog();
        final double nuAlpha = tree.getTreeSize();

        final double w = 1.0 / (m + t0);

        acceptRate = (alpha / nuAlpha);

        hBar = (1 - w) * hBar + w * (sigma - acceptRate);

        logStepSize = mu - (Math.sqrt(m) / gamma) * hBar;

        double tendToZero = Math.pow(m, -kappa);
        logStepSizeFrozen = tendToZero * logStepSize + (1 - tendToZero) * logStepSizeFrozen;

        return logStepSize;
    }

    public double getStepSize() {
        return stepsize;
    }

    @Override
    public void save(Statistics statistics) {
        statistics.store(NUTS.Metrics.STEPSIZE, stepsize);
        statistics.store(NUTS.Metrics.MEAN_TREE_ACCEPT, acceptRate);
    }
}
