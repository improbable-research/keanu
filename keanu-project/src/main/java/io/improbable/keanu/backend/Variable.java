package io.improbable.keanu.backend;

public interface Variable<T> {

    VariableReference getReference();

    T getValue();

    long[] getShape();
}
