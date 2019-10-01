package io.improbable.keanu.algorithms.variational.optimizer.gradient;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessAndGradient;
import io.improbable.keanu.algorithms.variational.optimizer.FitnessFunctionGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.VariableTransform;
import lombok.AllArgsConstructor;

import java.util.HashMap;
import java.util.Map;

@AllArgsConstructor
public class ReparameterizationAdapter implements FitnessFunctionGradient {

    private final FitnessFunctionGradient fitnessFunctionGradient;
    private final Map<VariableReference, VariableTransform> transforms;

    @Override
    public Map<VariableReference, DoubleTensor> getGradientsAt(Map<VariableReference, DoubleTensor> values) {
        return transformGradients(fitnessFunctionGradient.getGradientsAt(transform(values)));
    }

    @Override
    public FitnessAndGradient getFitnessAndGradientsAt(Map<VariableReference, DoubleTensor> values) {
        final FitnessAndGradient fitnessAndGradientsAt = fitnessFunctionGradient.getFitnessAndGradientsAt(transform(values));
        return new FitnessAndGradient(fitnessAndGradientsAt.getFitness(), transformGradients(fitnessAndGradientsAt.getGradients()));
    }

    private Map<VariableReference, DoubleTensor> transformGradients(final Map<VariableReference, DoubleTensor> dLogProbWrt) {
        transforms.forEach((id, t) -> dLogProbWrt.put(id, t.dTransform(dLogProbWrt.get(id))));
        return dLogProbWrt;
    }

    @Override
    public double getFitnessAt(Map<VariableReference, DoubleTensor> values) {
        return fitnessFunctionGradient.getFitnessAt(transform(values));
    }

    public Map<VariableReference, DoubleTensor> transform(Map<VariableReference, DoubleTensor> input) {
        Map<VariableReference, DoubleTensor> result = new HashMap<>(input);
        transforms.forEach((id, t) -> result.put(id, t.transform(result.get(id))));
        return result;
    }
}
