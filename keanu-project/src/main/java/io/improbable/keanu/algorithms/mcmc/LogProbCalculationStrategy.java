package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public interface LogProbCalculationStrategy {
    double calculate(ProbabilisticModel model, Set<Variable> variables);
}
