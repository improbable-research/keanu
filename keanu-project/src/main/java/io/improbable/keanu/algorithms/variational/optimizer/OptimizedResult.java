package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class OptimizedResult {

    final Map<VariableReference, DoubleTensor> optimizedValues;

    @Getter
    final double optimizedFitness;

    public DoubleTensor get(VariableReference variableReference) {
        return optimizedValues.get(variableReference);
    }
}