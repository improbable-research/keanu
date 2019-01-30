package io.improbable.keanu.algorithms.variational.optimizer.nongradient.testcase;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;

import java.util.List;

public interface NonGradientOptimizationAlgorithmTestCase {

    FitnessFunction getFitnessFunction();

    List<? extends Variable> getVariables();

    void assertResult(OptimizedResult result);
}
