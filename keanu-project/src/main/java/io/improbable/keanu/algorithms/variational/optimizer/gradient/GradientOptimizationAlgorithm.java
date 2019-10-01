package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.OptimizedResult;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.List;

public interface GradientOptimizationAlgorithm {

    OptimizedResult optimize(final List<? extends Variable<DoubleTensor, ?>> latentVariables,
                             FitnessFunctionGradient fitnessFunctionGradient);
}
