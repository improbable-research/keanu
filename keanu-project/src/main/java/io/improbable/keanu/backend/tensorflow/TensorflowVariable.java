package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.tensor.Tensor;
import lombok.AllArgsConstructor;


@AllArgsConstructor
public class TensorflowVariable<T> implements Variable<T> {

    private final TensorflowComputableGraph graph;
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
}
