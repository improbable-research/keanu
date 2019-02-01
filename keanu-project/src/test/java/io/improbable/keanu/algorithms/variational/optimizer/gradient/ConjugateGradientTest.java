package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import org.junit.Test;

public class ConjugateGradientTest {

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnNegativeMaxIterations() {
        validateParameters(-10, 0.1, 0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnNegativeRelativeThreshold() {
        validateParameters(10, -0.1, 0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnNegativeAbsoluteThreshold() {
        validateParameters(10, 0.1, -0.1);
    }

    public void validateParameters(int maxIterations, double relativeThreshold, double absoluteThreshold) {

        ConjugateGradient.builder()
            .maxEvaluations(maxIterations)
            .relativeThreshold(relativeThreshold)
            .absoluteThreshold(absoluteThreshold)
            .build();
    }
}
