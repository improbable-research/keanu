package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.algorithms.variational.optimizer.ConvergenceChecker;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilityFitness;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.GradientOptimizationAlgorithmTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.HimmelblauTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.RosenbrockTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.SingleGaussianTestCase;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase.SumGaussianTestCase;
import lombok.AllArgsConstructor;
import org.junit.Rule;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

import java.util.function.Supplier;

@RunWith(Theories.class)
public class GradientOptimizationAlgorithmTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @AllArgsConstructor
    public enum OptimizerType {

        ADAM(() -> {

            return Adam.builder()
                .maxEvaluations(100000)
                .alpha(0.1)
                .convergenceChecker(ConvergenceChecker.absoluteChecker(ConvergenceChecker.Norm.L2, 0.00001))
                .build();
        }),

        CONJUGATE_GRADIENT_POLAK_RIBIERE(() -> {

            return ConjugateGradient.builder()
                .updateFormula(ConjugateGradient.UpdateFormula.POLAK_RIBIERE)
                .build();
        }),


        CONJUGATE_GRADIENT_FLETCHER_REEVES(() -> {

            return ConjugateGradient.builder()
                .updateFormula(ConjugateGradient.UpdateFormula.FLETCHER_REEVES)
                .build();
        });

        private Supplier<GradientOptimizationAlgorithm> getOptimizer;
    }

    @AllArgsConstructor
    public enum TestCase {
        SINGLE_GAUSSIAN(SingleGaussianTestCase::new),
        SUM_GAUSSIAN_MAP(() -> new SumGaussianTestCase(ProbabilityFitness.MAP)),
        SUM_GAUSSIAN_MLE(() -> new SumGaussianTestCase(ProbabilityFitness.MLE)),
        ROSENBROCK_1_100(() -> new RosenbrockTestCase(1, 100)),
        HIMMELBLAU_A(() -> new HimmelblauTestCase(0, 3)),
        HIMMELBLAU_B(() -> new HimmelblauTestCase(-2, -3));

        private Supplier<GradientOptimizationAlgorithmTestCase> supplier;
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

        GradientOptimizationAlgorithmTestCase testCase = testCaseSupplier.supplier.get();

        GradientOptimizationAlgorithm algo = type.getOptimizer.get();

        testCase.assertUsingOptimizer(algo);
    }
}
