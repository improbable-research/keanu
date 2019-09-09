package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilityFitness;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.GradientOptimizationAlgorithmTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.HimmelblauTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.RosenbrockTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.SingleGaussianTestCase;
import org.junit.Test;

public class LBFGSTest {

    @Test
    public void canOptimizeSingleGaussianVectorNetwork() {
        assertPassesTestCase(new SingleGaussianTestCase());
    }

    @Test
    public void canOptimizeSingleGaussianNetwork() {
        assertPassesTestCase(new SingleGaussianTestCase(ProbabilityFitness.MAP, new long[0]));
    }

    @Test
    public void canOptimizeRosenbrock() {
        assertPassesTestCase(new RosenbrockTestCase(1, 100));
    }

    @Test
    public void canOptimizeHimmelblau() {
        assertPassesTestCase(new HimmelblauTestCase(0, 3));
        assertPassesTestCase(new HimmelblauTestCase(-2, -3));
    }

    private void assertPassesTestCase(GradientOptimizationAlgorithmTestCase testCase) {
        LBFGS lbfgs = LBFGS.builder().build();

        OptimizedResult result = lbfgs.optimize(
            testCase.getVariables(),
            testCase.getFitnessFunction(),
            testCase.getFitnessFunctionGradient()
        );

        testCase.assertResult(result);
    }
}
