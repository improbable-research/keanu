package io.improbable.keanu.algorithms.variational.optimizer;

public interface Variable<T> {

    VariableReference getReference();

    T getValue();

    void setValue(T tensor);

    long[] getShape();
}
