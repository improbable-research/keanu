package io.improbable.keanu.algorithms.mcmc;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;

public class Stepsize {

    private static final double STABILISER = 10;
    private static final double SHRINKAGE_FACTOR = 0.05;
    private static final double TEND_TO_ZERO_EXPONENT = 0.75;
    private static final double STARTING_STEPSIZE = 1;

    public double stepsize;

    private double averageAcceptanceProb;
    private double targetAcceptanceProb;
    private double logStepSizeFrozen;
    private double adaptCount;
    private double shrinkageTarget;

    Stepsize(double stepsize, double targetAcceptanceProb, int adaptCount) {
        this.averageAcceptanceProb = 0;
        this.targetAcceptanceProb = targetAcceptanceProb;
        this.stepsize = STARTING_STEPSIZE;
        this.logStepSizeFrozen = Math.log(stepsize);
        this.adaptCount = adaptCount;
        this.shrinkageTarget = Math.log(10 * stepsize);
    }

    /**
     * Taken from algorithm 4 in https://arxiv.org/pdf/1111.4246.pdf.
     */
    public static double findStartingStepSize(Map<VertexId, DoubleTensor> position,
                                       Map<VertexId, DoubleTensor> gradient,
                                       List<Vertex<DoubleTensor>> vertices,
                                       List<Vertex> probabilisticVertices,
                                       LogProbGradientCalculator logProbGradientCalculator,
                                       double initialLogOfMasterP,
                                       KeanuRandom random) {
        double stepsize = STARTING_STEPSIZE;
        Map<VertexId, DoubleTensor> momentums = new HashMap<>();
        initializeMomentumForEachVertex(vertices, momentums, random);

        Leapfrog leapfrog = new Leapfrog(position, momentums, gradient);
        double pThetaR = initialLogOfMasterP - leapfrog.halfDotProductMomentum();

        Leapfrog delta = leapfrog.step(vertices, logProbGradientCalculator, STARTING_STEPSIZE);

        double probAfterLeapfrog = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);
        double pThetaRAfterLeapFrog = probAfterLeapfrog - delta.halfDotProductMomentum();

        double logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        double scalingFactor = logLikelihoodRatio > Math.log(0.5) ? 1 : -1;

        while (scalingFactor * logLikelihoodRatio > -scalingFactor * Math.log(2)) {
            stepsize = stepsize * Math.pow(2, scalingFactor);

            delta = leapfrog.step(vertices, logProbGradientCalculator, stepsize);
            probAfterLeapfrog = ProbabilityCalculator.calculateLogProbFor(probabilisticVertices);
            pThetaRAfterLeapFrog = probAfterLeapfrog - delta.halfDotProductMomentum();

            logLikelihoodRatio = pThetaRAfterLeapFrog - pThetaR;
        }


        return stepsize;
    }

    public double adaptStepSize(TreeBuilder tree, int sampleNum) {

        if (sampleNum < adaptCount) {

            //1/(m+t0)
            double percentageLeftToTune = (1 / (sampleNum + STABILISER));

            //(1 - 1/(m+t0)) * Hm-1
            double proportionalAcceptanceProb = (1 - percentageLeftToTune) * averageAcceptanceProb;

            //alpha/nu_alpha
            double averageDeltaLikelihoodLeapfrog = tree.deltaLikelihoodOfLeapfrog / tree.treeSize;

            //delta - alpha/nu_alpha
            double acceptanceProb = targetAcceptanceProb - averageDeltaLikelihoodLeapfrog;

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

            //log(epsilon_bar_m) = m^-k * log(epsilon_m) + (1 - m^-k) * log(epsilon_bar_m-1)
            logStepSizeFrozen = reducedStepSize + increasedStepSizeFrozen;

            averageAcceptanceProb = updatedAverageAcceptanceProb;
            stepsize = Math.exp(updatedLogStepSize);
            return stepsize;
        } else {
            stepsize = Math.exp(logStepSizeFrozen);
            return stepsize;
        }
    }

    private static void initializeMomentumForEachVertex(List<Vertex<DoubleTensor>> vertices,
                                                        Map<VertexId, DoubleTensor> momentums,
                                                        KeanuRandom random) {
        for (Vertex<DoubleTensor> vertex : vertices) {
            momentums.put(vertex.getId(), random.nextGaussian(vertex.getShape()));
        }
    }

}

