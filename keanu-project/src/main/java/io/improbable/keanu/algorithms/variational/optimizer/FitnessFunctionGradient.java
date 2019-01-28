package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;

import java.util.Map;
import java.util.function.BiConsumer;

@AllArgsConstructor
public class FitnessFunctionGradient {

    private final ProbabilisticWithGradientGraph probabilisticWithGradientGraph;
    private final boolean useLikelihood;

    private final BiConsumer<Map<VariableReference, DoubleTensor>, Map<? extends VariableReference, DoubleTensor>> onGradientCalculation;

    public Map<? extends VariableReference, DoubleTensor> value(Map<VariableReference, DoubleTensor> values) {

        Map<? extends VariableReference, DoubleTensor> diffs = useLikelihood ?
            probabilisticWithGradientGraph.logLikelihoodGradients(values) :
            probabilisticWithGradientGraph.logProbGradients(values);

        if (onGradientCalculation != null) {
            onGradientCalculation.accept(values, diffs);
        }

        return diffs;
    }

}
