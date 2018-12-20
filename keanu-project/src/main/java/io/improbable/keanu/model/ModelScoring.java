package io.improbable.keanu.model;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.experimental.UtilityClass;

@UtilityClass
public class ModelScoring {
    /**
     * Calculates the <a href="https://en.wikipedia.org/wiki/Coefficient_of_determination">coefficient of determination</a> - i.e.
     * the R^2 value - for the given predicted output and true values.
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


    /**
     * Calculates the accuracy of a predicted output, given a true output. That is the ratio of correct predictions to
     * the total number of predictions.
     *
     * @param predictedOutput predicted output
     * @param trueOutput      the true output to compare against
     * @return the accuracy
     */
    public static double accuracy(BooleanTensor predictedOutput, BooleanTensor trueOutput) {
        return predictedOutput.elementwiseEquals(trueOutput).toDoubleMask().sum() / trueOutput.getLength();
    }
}
