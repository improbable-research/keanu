package io.improbable.keanu.algorithms.mcmc;

public class AutoTune {

    private static final double STABILISER = 10;
    private static final double SHRINKAGE_FACTOR = 0.05;
    private static final double TEND_TO_ZERO_EXPONENT = 0.75;

    private double averageAcceptanceProb;
    private double targetAcceptanceProb;
    private double logStepSizeFrozen;
    private double adaptCount;
    private double shrinkageTarget;

    AutoTune(double logStepSize, double targetAcceptanceProb, int adaptCount) {
        this.averageAcceptanceProb = 0;
        this.targetAcceptanceProb = targetAcceptanceProb;
        this.logStepSizeFrozen = Math.log(1);
        this.adaptCount = adaptCount;
        this.shrinkageTarget = Math.log(10 * Math.exp(logStepSize));
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
            return Math.exp(updatedLogStepSize);
        } else {

            return Math.exp(logStepSizeFrozen);
        }
    }
}

