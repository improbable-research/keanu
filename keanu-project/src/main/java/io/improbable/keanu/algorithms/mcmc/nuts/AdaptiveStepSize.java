package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.List;


/**
 * This is based on Algorithm 5 from https://arxiv.org/pdf/1111.4246.pdf.
 * <p>
 * The step size is used by NUTS as an epsilon for the leap frog integration.
 */
class AdaptiveStepSize {

    private static final double t0 = 10;
    private static final double gamma = 0.05;
    private static final double kappa = 0.75;

    private final double mu;
    private final long adaptCount;
    private final double sigma;

    private double stepSize;
    private double hBar;
    private double acceptRate;

    private double logStepSizeBar;
    private double logStepSize;

    private long stepNum;

    /**
     * @param stepSize   the step size
     * @param sigma      the target acceptance probability (lower target equates to a higher step size when tuning)
     * @param adaptCount the number of samples to adapt for
     */
    AdaptiveStepSize(double stepSize, double sigma, long adaptCount) {
        this.sigma = sigma;
        this.stepSize = stepSize;
        this.hBar = 0;
        this.logStepSize = Math.log(stepSize);
        this.logStepSizeBar = logStepSize;
        this.adaptCount = adaptCount;
        this.mu = Math.log(10 * stepSize);
        this.stepNum = 1;
    }

    public static double findStartingStepSize(double stepScale, List<? extends Variable<DoubleTensor, ?>> variables) {
        long N = variables.stream()
            .mapToLong(v -> v.getValue().getLength())
            .sum();

        return stepScale / Math.pow(N, 0.25);
    }

    /**
     * Adapts the step size based on the state of the tree and network after computing a sample
     *
     * @param tree the balanced binary tree
     * @return a new step size
     */
    public double adaptStepSize(Tree tree) {

        if (stepNum <= adaptCount) {

            stepSize = Math.exp(updateLogStepSize(tree));

        } else if (stepNum == adaptCount + 1) {

            stepSize = Math.exp(logStepSizeBar);
        }

        stepNum++;

        return stepSize;
    }

    private double updateLogStepSize(Tree tree) {

        final double alpha = tree.getSumMetropolisAcceptanceProbability();
        final double nuAlpha = tree.getTreeSize();

        final double w = 1.0 / (stepNum + t0);

        acceptRate = (alpha / nuAlpha);

        hBar = (1 - w) * hBar + w * (sigma - acceptRate);

        logStepSize = mu - (Math.sqrt(stepNum) / gamma) * hBar;

        final double tendToZero = Math.pow(stepNum, -kappa);
        logStepSizeBar = tendToZero * logStepSize + (1 - tendToZero) * logStepSizeBar;

        return logStepSize;
    }

    public double getStepSize() {
        return stepSize;
    }

    public void save(Statistics statistics) {
        statistics.store(NUTS.Metrics.STEPSIZE, stepSize);
        statistics.store(NUTS.Metrics.MEAN_TREE_ACCEPT, acceptRate);
    }
}
