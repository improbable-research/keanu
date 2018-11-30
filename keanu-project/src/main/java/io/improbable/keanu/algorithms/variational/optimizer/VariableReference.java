package io.improbable.keanu.algorithms.variational.optimizer;

public interface VariableReference {

    Object getValue();

    long[] getShape();
}
