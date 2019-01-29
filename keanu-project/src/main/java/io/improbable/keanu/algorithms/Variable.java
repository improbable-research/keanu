package io.improbable.keanu.algorithms;

import io.improbable.keanu.network.VariableState;

public interface Variable<VALUE, STATE extends VariableState> {

    VariableReference getReference();

    VALUE getValue();

    long[] getShape();

    STATE getState();
}
