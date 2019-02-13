package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;

import java.util.List;

public interface NonGradientOptimizationAlgorithm {

    OptimizedResult optimize(final List<? extends Variable> latentVariables,
                             FitnessFunction fitnessFunction);
}
