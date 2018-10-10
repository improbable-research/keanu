package io.improbable.keanu.model;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.experimental.UtilityClass;

@UtilityClass
public class LinearModelScore {
    /**
     * Calculates the <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination">coefficient of determination</a> - i.e.
     * the R² value - for the given predicted output and true values.
     *
     * @param predictedOutput predicted output
     * @param trueOutput      the true output to compare the predicted output against
     * @return the coefficient, a value less than 1, where any value above 0 resembles correlation
     */
    public static double coefficientOfDetermination(DoubleTensor predictedOutput, DoubleTensor trueOutput) {
        double residualSumOfSquares = (trueOutput.minus(predictedOutput).pow(2.)).sum();
        double totalSumOfSquares = ((trueOutput.minus(trueOutput.average())).pow(2.)).sum();
        return 1 - (residualSumOfSquares / totalSumOfSquares);
    }
}
