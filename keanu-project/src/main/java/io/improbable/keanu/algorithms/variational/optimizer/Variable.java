package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.network.VariableState;
import io.improbable.keanu.vertices.Samplable;

public interface Variable<VALUE, STATE extends VariableState> {

    VariableReference getReference();

    VALUE getValue();

    void setValue(VALUE value);

    long[] getShape();

    STATE getState();

    void setState(STATE state);
}
