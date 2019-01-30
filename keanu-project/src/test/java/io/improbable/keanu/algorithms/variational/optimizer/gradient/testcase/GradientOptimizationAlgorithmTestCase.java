package io.improbable.keanu.algorithms.variational.optimizer.gradient.testcase;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;

import java.util.List;

public interface GradientOptimizationAlgorithmTestCase {

    FitnessFunction getFitnessFunction();

    FitnessFunctionGradient getFitnessFunctionGradient();

    List<? extends Variable> getVariables();

    void assertResult(OptimizedResult result);
}
