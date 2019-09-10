package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessAndGradient;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.ProbabilityFitnessFunction;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Map;
import java.util.function.BiConsumer;

public abstract class ProbabilityFitnessFunctionGradient extends ProbabilityFitnessFunction implements FitnessFunctionGradient {

    private final ProbabilisticModelWithGradient probabilisticModelWithGradient;

    private final BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation;

    public ProbabilityFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient) {
        super(probabilisticModelWithGradient);
        this.probabilisticModelWithGradient = probabilisticModelWithGradient;
        this.onGradientCalculation = (point, gradient) -> {
        };
    }

    public ProbabilityFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                              BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation,
                                              BiConsumer<Map<VariableReference, DoubleTensor>, Double> onFitnessCalculation) {
        super(probabilisticModelWithGradient, onFitnessCalculation);
        this.probabilisticModelWithGradient = probabilisticModelWithGradient;
        this.onGradientCalculation = onGradientCalculation;
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> getGradientsAt(Map<VariableReference, DoubleTensor> values) {

        final Map<? extends VariableReference, DoubleTensor> gradients = calculateGradients(probabilisticModelWithGradient, values);

        onGradientCalculation.accept(values, gradients);

        return gradients;
    }

    @Override
    public FitnessAndGradient getFitnessAndGradientsAt(Map<VariableReference, DoubleTensor> values) {

        FitnessAndGradient fitnessAndGradient = calculateFitnessAndGradientsAt(probabilisticModelWithGradient, values);

        onFitnessCalculation.accept(values, fitnessAndGradient.getFitness());
        onGradientCalculation.accept(values, fitnessAndGradient.getGradients());

        return fitnessAndGradient;
    }

    protected abstract Map<? extends VariableReference, DoubleTensor> calculateGradients(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                                                         Map<VariableReference, DoubleTensor> values);

    protected abstract FitnessAndGradient calculateFitnessAndGradientsAt(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                                         Map<VariableReference, DoubleTensor> values);
}
