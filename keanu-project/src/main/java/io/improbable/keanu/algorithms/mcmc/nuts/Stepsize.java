package io.improbable.keanu.algorithms.mcmc.nuts;

import io.improbable.keanu.algorithms.SaveStatistics;
import io.improbable.keanu.algorithms.Statistics;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


/**
 * Used by NUTS as an epsilon for the leap frog
 */
class Stepsize implements SaveStatistics {

    private static final double STABILISER = 10;
    private static final double SHRINKAGE_FACTOR = 0.05;
    private static final double TEND_TO_ZERO_EXPONENT = 0.75;
    private static final double STARTING_STEPSIZE = 1;

    private final double shrinkageTarget;
    private final double adaptCount;
    private final double targetAcceptanceProb;

    private double stepsize;
    private double averageAcceptanceProb;
    private double logStepSizeFrozen;

    /**
     * @param stepsize             the step size
     * @param targetAcceptanceProb the target acceptance probability (lower target equates to a higher step size when tuning)
     * @param adaptCount           the number of samples to adapt for
     */
    Stepsize(double stepsize, double targetAcceptanceProb, int adaptCount) {
        this.targetAcceptanceProb = targetAcceptanceProb;
        this.stepsize = stepsize;
        this.logStepSizeFrozen = Math.log(stepsize);
        this.adaptCount = adaptCount;
        this.shrinkageTarget = Math.log(10 * stepsize);
    }

    /**
     * Taken from algorithm 4 in https://arxiv.org/pdf/1111.4246.pdf.
     *
     * @param position                  the starting position
     * @param gradient                  the gradient at the starting position
     * @param variables                  the variables
     * @param logProbGradientCalculator the log prob gradient calculator
     * @param initialLogOfMasterP       the initial master log prob
     * @param random                    the source of randomness
     * @return a starting step size
     */
    public static double findStartingStepSize(Map<VariableReference, DoubleTensor> position,
                                              Map<? extends VariableReference, DoubleTensor> gradient,
                                              List<? extends Variable<DoubleTensor>> variables,
                                              ProbabilisticModelWithGradient logProbGradientCalculator,
                                              double initialLogOfMasterP,
                                              KeanuRandom random) {
        double stepsize = STARTING_STEPSIZE;
        Map<VariableReference, DoubleTensor> momentums = variables.stream()
            .collect(Collectors.toMap(
                Variable::getReference,
                v -> random.nextGaussian(v.getShape())
            ));

        Leapfrog leapfrog = new Leapfrog(position, momentums, gradient);
        double pThetaR = initialLogOfMasterP - leapfrog.halfDotProductMomentum();

        Leapfrog delta = leapfrog.step(variables, logProbGradientCalculator, STARTING_STEPSIZE);

        double probAfterLeapfrog = logProbGradientCalculator.logProb();
        double pThetaRAfterLeapFrog = probAfterLeapfrog - delta.halfDotProductMomentum();

        double logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        double scalingFactor = logLikelihoodRatio > Math.log(0.5) ? 1 : -1;

        while (scalingFactor * logLikelihoodRatio > -scalingFactor * Math.log(2)) {
            stepsize = stepsize * Math.pow(2, scalingFactor);

            delta = leapfrog.step(variables, logProbGradientCalculator, stepsize);
            probAfterLeapfrog = logProbGradientCalculator.logProb();
            pThetaRAfterLeapFrog = probAfterLeapfrog - delta.halfDotProductMomentum();

            logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        }

        return stepsize;
    }

    /**
     * Adapts the step size based on the state of the tree and network after computing a sample
     *
     * @param tree      the balanced binary tree
     * @param sampleNum the number of samples that have been taken
     * @return a new step size
     */
    public double adaptStepSize(Tree tree, int sampleNum) {

        double logStepSize = logStepSizeFrozen;

        if (sampleNum < adaptCount) {
            logStepSize = updateLogStepSize(tree, sampleNum);
        }

        stepsize = Math.exp(logStepSize);
        return stepsize;
    }

    private double updateLogStepSize(Tree tree, int sampleNum) {

        //1/(m+t0)
        double percentageLeftToTune = (1 / (sampleNum + STABILISER));

        //(1 - 1/(m+t0)) * Hm-1
        double proportionalAcceptanceProb = (1 - percentageLeftToTune) * averageAcceptanceProb;

        //alpha/nu_alpha
        double averageTreeAcceptanceProb = tree.getDeltaLikelihoodOfLeapfrog() / tree.getTreeSize();

        //delta - alpha/nu_alpha
        double acceptanceProb = targetAcceptanceProb - averageTreeAcceptanceProb;

        //Hm = (1-1/(m+t0)) * Hm-1 + (1/(m+t0)) * (delta - (alpha/nu_alpha))
        double updatedAverageAcceptanceProb = proportionalAcceptanceProb + (percentageLeftToTune * acceptanceProb);

        //sqrt(mu)/gamma
        double shrunkSampleCount = Math.sqrt(sampleNum) / SHRINKAGE_FACTOR;

        //log(epsilon_m) = mu - (sqrt(m)/gamma) * Hm
        double updatedLogStepSize = shrinkageTarget - (shrunkSampleCount * updatedAverageAcceptanceProb);

        //m^-k
        double tendToZero = Math.pow(sampleNum, -TEND_TO_ZERO_EXPONENT);

        //m^-k * log(epsilon_m)
        double reducedStepSize = tendToZero * updatedLogStepSize;

        //(1-m^-k) * log(epsilon_bar_m-1)
        double increasedStepSizeFrozen = (1 - tendToZero) * logStepSizeFrozen;

        averageAcceptanceProb = updatedAverageAcceptanceProb;

        //log(epsilon_bar_m) = m^-k * log(epsilon_m) + (1 - m^-k) * log(epsilon_bar_m-1)
        return reducedStepSize + increasedStepSizeFrozen;
    }

    public double getStepsize() {
        return stepsize;
    }

    @Override
    public void save(Statistics statistics) {
        statistics.store(NUTS.Metrics.STEPSIZE, stepsize);
        statistics.store(NUTS.Metrics.MEAN_TREE_ACCEPT, averageAcceptanceProb);
    }
}
