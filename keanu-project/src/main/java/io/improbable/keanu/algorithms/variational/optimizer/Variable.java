package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.VariableState;
import io.improbable.keanu.vertices.Samplable;

public interface Variable<T> extends Samplable<T> {

    VariableReference getReference();

    T getValue();

    void setValue(T tensor);

    long[] getShape();

    VariableState getState();

    void setState(VariableState variableState);
}
