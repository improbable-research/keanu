package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.ProbabilisticModel;
import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessAndGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;
import java.util.function.BiConsumer;

public class LogLikelihoodFitnessFunctionGradient extends ProbabilityFitnessFunctionGradient {

    public LogLikelihoodFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation,
                                                BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation) {
        super(probabilisticModelWithGradient, onGradientCalculation, onFitnessCalculation);
    }

    public LogLikelihoodFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient) {
        super(probabilisticModelWithGradient);
    }

    @Override
    protected double calculateFitness(ProbabilisticModel probabilisticModel, Map<VariableReference, DoubleTensor> values) {
        return probabilisticModel.logLikelihood(values);
    }

    @Override
    protected Map<VariableReference, DoubleTensor> calculateGradients(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                                                Map<VariableReference, DoubleTensor> values) {
        return probabilisticModelWithGradient.logLikelihoodGradients(values);
    }

    @Override
    protected FitnessAndGradient calculateFitnessAndGradientsAt(ProbabilisticModelWithGradient probabilisticModelWithGradient, Map<VariableReference, DoubleTensor> values) {
        double fitness = probabilisticModelWithGradient.logLikelihood(values);
        Map<VariableReference, DoubleTensor> gradients = probabilisticModelWithGradient.logLikelihoodGradients();

        return new FitnessAndGradient(fitness, gradients);
    }
}