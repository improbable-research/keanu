package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.ProbabilisticModelWithGradient;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public abstract class ProbabilityFitnessFunctionGradient implements FitnessFunctionGradient {

    private final ProbabilisticModelWithGradient probabilisticModelWithGradient;

    private final BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation;

    public ProbabilityFitnessFunctionGradient(ProbabilisticModelWithGradient probabilisticModelWithGradient) {
        this(probabilisticModelWithGradient, (point, gradient) -> {
        });
    }

    @Override
    public Map<? extends VariableReference, DoubleTensor> getGradientsAt(Map<VariableReference, DoubleTensor> values) {

        final Map<? extends VariableReference, DoubleTensor> gradients = calculateGradients(probabilisticModelWithGradient, values);

        onGradientCalculation.accept(values, gradients);

        return gradients;
    }

    abstract Map<? extends VariableReference, DoubleTensor> calculateGradients(ProbabilisticModelWithGradient probabilisticModelWithGradient,
                                                                               Map<VariableReference, DoubleTensor> values);
}
