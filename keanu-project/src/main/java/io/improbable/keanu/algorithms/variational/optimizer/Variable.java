package io.improbable.keanu.algorithms.variational.optimizer;

public interface Variable<T> extends HasShape {

    VariableReference getReference();

    T getValue();

    long[] getShape();
}
