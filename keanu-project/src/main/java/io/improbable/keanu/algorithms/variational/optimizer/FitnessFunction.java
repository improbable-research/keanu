package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public class FitnessFunction {

    private final ProbabilisticGraph probabilisticGraph;
    private final boolean useLikelihood;

    private final BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation;

    public double value(Map<VariableReference, DoubleTensor> values) {

        double logOfTotalProbability = useLikelihood ?
            probabilisticGraph.logLikelihood(values) :
            probabilisticGraph.logProb(values);

        if (onFitnessCalculation != null) {
            onFitnessCalculation.accept(values, logOfTotalProbability);
        }

        return logOfTotalProbability;
    }

}
