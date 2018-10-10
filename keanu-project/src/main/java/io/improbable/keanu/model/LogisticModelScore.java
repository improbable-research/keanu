package io.improbable.keanu.model;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import lombok.experimental.UtilityClass;

@UtilityClass
public class LogisticModelScore {

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
