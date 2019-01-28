package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.SingleGaussianTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.SumGaussianTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.testcase.NonGradientOptimizationAlgorithmTestCase;
import lombok.AllArgsConstructor;
import org.junit.Rule;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

import java.util.function.Supplier;

@RunWith(Theories.class)
public class NonGradientOptimizationAlgorithmTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @AllArgsConstructor
    public enum OptimizerType {

        BOBYQA_ALGO(() -> {

            return BOBYQA.builder()
                .build();
        });

        Supplier<NonGradientOptimizationAlgorithm> getOptimizer;
    }

    @AllArgsConstructor
    public enum TestCase {
        SUM_GAUSSIAN_MAP(() -> new SumGaussianTestCase(true)),
        SUM_GAUSSIAN_MLE(() -> new SumGaussianTestCase(false)),
        SINGLE_GAUSSIAN(SingleGaussianTestCase::new);

        private Supplier<NonGradientOptimizationAlgorithmTestCase> supplier;
    }

    @DataPoints
    public static OptimizerType[] getTypes() {
        return OptimizerType.values();
    }

    @DataPoints
    public static TestCase[] getTestCase() {
        return TestCase.values();
    }

    @Theory
    public void canOptimize(OptimizerType type, TestCase testCaseSupplier) {

        NonGradientOptimizationAlgorithmTestCase testCase = testCaseSupplier.supplier.get();

        NonGradientOptimizationAlgorithm algo = type.getOptimizer.get();

        OptimizedResult result = algo.optimize(
            testCase.getVariables(),
            testCase.getFitnessFunction()
        );

        testCase.assertResult(result);
    }
}