package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilityFitness;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.GradientOptimizationAlgorithmTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.SingleGaussianTestCase;
import org.apache.commons.lang3.mutable.MutableInt;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

public class AdamTest {

    @Test
    public void canOptimizeSingleGaussianNetwork() {

        GradientOptimizationAlgorithmTestCase testCase = new SingleGaussianTestCase(ProbabilityFitness.MAP, new long[0]);

        Adam adamOptimizer = Adam.builder()
            .build();

        OptimizedResult result = adamOptimizer.optimize(
            testCase.getVariables(),
            testCase.getFitnessFunction(),
            testCase.getFitnessFunctionGradient()
        );

        testCase.assertResult(result);
    }

    @Test
    public void canOptimizeSingleGaussianVectorNetwork() {

        SingleGaussianTestCase testCase = new SingleGaussianTestCase();

        Adam adamOptimizer = Adam.builder()
            .alpha(0.1)
            .build();

        OptimizedResult result = adamOptimizer.optimize(
            testCase.getVariables(),
            testCase.getFitnessFunction(),
            testCase.getFitnessFunctionGradient()
        );

        testCase.assertResult(result);
    }

    @Test
    public void canAddConvergenceChecker() {

        GradientOptimizationAlgorithmTestCase testCase = new SingleGaussianTestCase(ProbabilityFitness.MAP, new long[0]);

        MutableInt i = new MutableInt(0);
        Adam adamOptimizer = Adam.builder()
            .alpha(0.1)
            .convergenceChecker((theta, thetaNext) -> i.incrementAndGet() == 10)
            .build();

        adamOptimizer.optimize(
            testCase.getVariables(),
            testCase.getFitnessFunction(),
            testCase.getFitnessFunctionGradient()
        );

        assertTrue(i.getValue() == 10);
    }

    @Test
    public void canAddMaxIterations() {

        GradientOptimizationAlgorithmTestCase testCase = new SingleGaussianTestCase(ProbabilityFitness.MAP, new long[0]);

        MutableInt i = new MutableInt(0);
        Adam adamOptimizer = Adam.builder()
            .maxEvaluations(5)
            .convergenceChecker((theta, thetaNext) -> {
                i.incrementAndGet();
                return false;
            })
            .alpha(0.1)
            .build();

        adamOptimizer.optimize(
            testCase.getVariables(),
            testCase.getFitnessFunction(),
            testCase.getFitnessFunctionGradient()
        );

        assertEquals(i.getValue(), new Integer(5));
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnNegativeBeta1() {
        validateParameters(10, 0.1, -0.1, 0.1, 0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnNegativeBeta2() {
        validateParameters(10, 0.1, 0.1, -0.1, 0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnBigBeta1() {
        validateParameters(10, 0.1, 1, 0.1, 0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnBigBeta2() {
        validateParameters(10, 0.1, 0.1, 1, 0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnNegativeAlpha() {
        validateParameters(10, -0.1, 0.1, 0.1, 0.1);
    }

    @Test(expected = IllegalArgumentException.class)
    public void throwsOnNegativeEpsilon() {
        validateParameters(10, 0.1, 0.1, 0.1, -0.1);
    }

    public void validateParameters(int maxIterations, double alpha, double beta1, double beta2, double epsilon) {

        Adam adamOptimizer = Adam.builder()
            .maxEvaluations(maxIterations)
            .alpha(alpha)
            .beta1(beta1)
            .beta2(beta2)
            .epsilon(epsilon)
            .build();
    }

}
