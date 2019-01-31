package io.improbable.keanu.algorithms.variational.optimizer.nongradient;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunction;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public abstract class ProbabilityFitnessFunction implements FitnessFunction {

    private final ProbabilisticModel probabilisticModel;
    private final BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation;

    public ProbabilityFitnessFunction(ProbabilisticModel probabilisticModel) {
        this(probabilisticModel, null);
    }

    @Override
    public double getFitnessAt(Map<VariableReference, DoubleTensor> values) {

        final double logProb = calculateFitness(probabilisticModel, values);

        if (onFitnessCalculation != null) {
            onFitnessCalculation.accept(values, logProb);
        }

        return logProb;
    }

    abstract double calculateFitness(ProbabilisticModel probabilisticModel,
                                     Map<VariableReference, DoubleTensor> values);
}
