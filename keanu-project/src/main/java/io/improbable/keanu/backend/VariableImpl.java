package io.improbable.keanu.backend;

import io.improbable.keanu.algorithms.Variable;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.Tensor;
import lombok.AllArgsConstructor;


@AllArgsConstructor
public class VariableImpl<T> implements Variable<T, VariableStateImpl> {

    private final ComputableGraph graph;
    private final VariableReference variableReference;

    @Override
    public VariableReference getReference() {
        return variableReference;
    }

    @Override
    public T getValue() {
        return graph.getInput(variableReference);
    }

    @Override
    public long[] getShape() {
        T value = getValue();
        if (value instanceof Tensor) {
            return ((Tensor) value).getShape();
        } else {
            return new long[0];
        }
    }

    @Override
    public VariableStateImpl getState() {
        return null;
    }
}
