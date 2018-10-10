package io.improbable.keanu.model.regression;

import java.util.Arrays;

import lombok.experimental.UtilityClass;

@UtilityClass
class RegressionWeights {
    private static void checkAmount(long desiredAmount, double[] amountToCheck, String name) {
        if (amountToCheck.length != desiredAmount) {
            throw new IllegalArgumentException(String.format("Expected %d %s, instead got %d", desiredAmount, name, amountToCheck.length));
        }
    }

    static void checkLaplaceParameters(long featureCount, double[] means, double[] betas) {
        checkAmount(featureCount, means, "means");
        checkAmount(featureCount, betas, "betas");
    }

    static void checkGaussianParameters(long featureCount, double[] means, double[] sigmas) {
        checkAmount(featureCount, means, "means");
        checkAmount(featureCount, sigmas, "sigmas");
    }

    static double[] fillPriorOnWeights(long[] featureShape, double weight) {
        double[] priorWeights = new double[Math.toIntExact(featureShape[0])];
        Arrays.fill(priorWeights, weight);
        return priorWeights;
    }
}
