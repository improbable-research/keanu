package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;

@AllArgsConstructor
public class OptimizedResult {

    private final Map<VariableReference, DoubleTensor> optimizedValues;

    @Getter
    private final double fitness;

    public DoubleTensor getValueFor(VariableReference variableReference) {
        return optimizedValues.get(variableReference);
    }
}