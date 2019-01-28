package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public class LogLikelihoodFitnessFunction implements FitnessFunction {

    private final ProbabilisticGraph probabilisticGraph;
    private final BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation;

    public LogLikelihoodFitnessFunction(ProbabilisticGraph probabilisticGraph) {
        this.probabilisticGraph = probabilisticGraph;
        this.onFitnessCalculation = null;
    }

    public double value(Map<VariableReference, DoubleTensor> values) {

        double logProb = probabilisticGraph.logLikelihood(values);

        if (onFitnessCalculation != null) {
            onFitnessCalculation.accept(values, logProb);
        }

        return logProb;
    }
}
