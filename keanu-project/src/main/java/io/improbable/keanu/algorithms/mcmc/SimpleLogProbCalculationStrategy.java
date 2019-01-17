package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticModel;
import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Set;

public class SimpleLogProbCalculationStrategy implements LogProbCalculationStrategy {
    @Override
    public double calculate(ProbabilisticModel model, Set<Variable> variables) {
        return model.logProb();
    }
}
