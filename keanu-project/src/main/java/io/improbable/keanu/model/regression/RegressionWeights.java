package io.improbable.keanu.model.regression;

import java.util.Arrays;

import lombok.experimental.UtilityClass;

@UtilityClass
class RegressionWeights {
    /**
     * Takes an array that contains the parameters for the distributions of weights in a regression model,
     * and checks that the number of parameters in the array matches the intended number of regression features.
     *
     * @param arrayToCheck The array to be checked
     * @param numberOfFeatures The number of features which the array being checked is expected to have
     */
    public static void checkArrayHasCorrectNumberOfFeatures(double[] arrayToCheck, long numberOfFeatures) {
        if (arrayToCheck.length != numberOfFeatures) {
            throw new IllegalArgumentException(String.format("Expected regression weights of length %d, instead got %d", numberOfFeatures, arrayToCheck.length));
        }
    }

    static double[] fillPriorOnWeights(long[] featureShape, double weight) {
        double[] priorWeights = new double[Math.toIntExact(featureShape[0])];
        Arrays.fill(priorWeights, weight);
        return priorWeights;
    }
}
