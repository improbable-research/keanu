package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessAndGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;
import java.util.function.BiConsumer;

public class LogProbFitnessFunctionGradient extends ProbabilityFitnessFunctionGradient {

    public LogProbFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                          BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation,
                                          BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation) {
        super(probabilisticModelWithGradient, onGradientCalculation, onFitnessCalculation);
    }

    public LogProbFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient) {
        super(probabilisticModelWithGradient);
    }

    @Override
    protected double calculateFitness(ProbabilisticModel probabilisticModel, Map<VariableReference, DoubleTensor> values) {
        return probabilisticModel.logProb(values);
    }

    @Override
    protected Map<? extends VariableReference, DoubleTensor> calculateGradients(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                                                Map<VariableReference, DoubleTensor> values) {
        return probabilisticModelWithGradient.logProbGradients(values);
    }

    @Override
    protected FitnessAndGradient calculateFitnessAndGradientsAt(ProbabilisticModelWithGradient probabilisticModelWithGradient, Map<VariableReference, DoubleTensor> values) {

        double fitness = probabilisticModelWithGradient.logProb(values);
        Map<VariableReference, DoubleTensor> gradients = probabilisticModelWithGradient.logProbGradients();

        return new FitnessAndGradient(fitness, gradients);
    }
}
