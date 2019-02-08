package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.SaveStatistics;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.List;


/**
 * Used by NUTS as an epsilon for the leap frog
 */
class Stepsize implements SaveStatistics {

    private static final double t0 = 10;
    private static final double gamma = 0.05;
    private static final double kappa = 0.75;

    private final double mu;
    private final double adaptCount;
    private final double sigma;

    private double stepSize;
    private double hBar;
    private double acceptRate;
    private double logStepSizeFrozen;
    private double logStepSize;

    /**
     * @param stepSize   the step size
     * @param sigma      the target acceptance probability (lower target equates to a higher step size when tuning)
     * @param adaptCount the number of samples to adapt for
     */
    Stepsize(double stepSize, double sigma, int adaptCount) {
        this.sigma = sigma;
        this.stepSize = stepSize;
        this.hBar = 0;
        this.logStepSize = Math.log(stepSize);
        this.logStepSizeFrozen = logStepSize;//Math.log(1);
        this.adaptCount = adaptCount;
        this.mu = Math.log(10 * stepSize);
    }

    public static double findStartingStepSizeSimple(double stepScale, List<? extends Variable<DoubleTensor, ?>> variables) {
        long N = variables.stream()
            .mapToLong(v -> v.getValue().getLength())
            .sum();

        return stepScale / Math.pow(N, 0.25);
    }

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

        stepSize = Math.exp(logStepSizeAtSample);
        return stepSize;
    }

    private double updateLogStepSize(Tree tree, int m) {

        final double alpha = tree.getAcceptSum();
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
        return stepSize;
    }

    @Override
    public void save(Statistics statistics) {
        statistics.store(NUTS.Metrics.STEPSIZE, stepSize);
        statistics.store(NUTS.Metrics.MEAN_TREE_ACCEPT, acceptRate);
    }
}
