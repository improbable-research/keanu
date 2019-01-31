package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.GradientOptimizationAlgorithmTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.SingleGaussianTestCase;
import org.apache.commons.lang3.mutable.MutableInt;
import org.junit.Test;

import static org.junit.Assert.assertTrue;

public class AdamTest {

    @Test
    public void canOptimizeSingleGaussianNetwork() {

        GradientOptimizationAlgorithmTestCase testCase = new SingleGaussianTestCase(false, new long[0]);

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

        GradientOptimizationAlgorithmTestCase testCase = new SingleGaussianTestCase(false, new long[0]);

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

}
