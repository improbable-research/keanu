package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public class SimpleLogProbCalculationStrategy implements LogProbCalculationStrategy {
    @Override
    public double calculate(ProbabilisticGraph graph, Set<Variable> variables) {
        return graph.logProb();
    }
}
