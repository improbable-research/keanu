package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.junit.Test;

public class ConjugateGradientTest {

    @Test(expected = NotStrictlyPositiveException.class)
    public void throwsOnNegativeMaxEvaluations() {
        validateParameters(-10, 0.1, 0.1);
    }

    @Test(expected = NotStrictlyPositiveException.class)
    public void throwsOnNegativeRelativeThreshold() {
        validateParameters(10, -0.1, 0.1);
    }

    @Test(expected = NotStrictlyPositiveException.class)
    public void throwsOnNegativeAbsoluteThreshold() {
        validateParameters(10, 0.1, -0.1);
    }

    public void validateParameters(int maxEvaluations, double relativeThreshold, double absoluteThreshold) {

        ConjugateGradient.builder()
            .maxEvaluations(maxEvaluations)
            .relativeThreshold(relativeThreshold)
            .absoluteThreshold(absoluteThreshold)
            .build();
    }
}
