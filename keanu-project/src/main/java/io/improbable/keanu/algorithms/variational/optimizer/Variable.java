package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.VariableState;

public interface Variable<T> {

    VariableReference getReference();

    T getValue();

    void setValue(T tensor);

    long[] getShape();

    VariableState getState();

    void setState(VariableState variableState);
}
