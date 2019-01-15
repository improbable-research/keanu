package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.optimizer.LambdaSectionSnapshot;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.vertices.Vertex;

import java.util.List;
import java.util.Set;

public class LambdaSectionOptimizedLogProbCalculator implements LogProbCalculationStrategy {

    private final LambdaSectionSnapshot lambdaSectionSnapshot;

    public LambdaSectionOptimizedLogProbCalculator(List<Vertex> latentVertices) {
        lambdaSectionSnapshot = new LambdaSectionSnapshot(latentVertices);
    }

    @Override
    public double calculate(ProbabilisticGraph graph, Set<Variable> variables) {
        return lambdaSectionSnapshot.logProb(variables);
    }
}
