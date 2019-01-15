package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public interface LogProbCalculationStrategy {
    double calculate(ProbabilisticGraph graph, Set<Variable> variables);
}
