package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;

import java.util.List;

public interface GradientOptimizationAlgorithm {

    OptimizedResult optimize(final List<? extends Variable> latentVariables,
                             FitnessFunction fitnessFunction,
                             FitnessFunctionGradient fitnessFunctionGradient);
}
