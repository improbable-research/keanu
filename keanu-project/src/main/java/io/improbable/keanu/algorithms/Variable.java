package io.improbable.keanu.algorithms;

import io.improbable.keanu.network.VariableState;

/**
 * @param <VALUE> A Variable must have a value - typically a {@link io.improbable.keanu.tensor.Tensor}
 * @param <STATE> A Variable's State can include other fields - for example {@link io.improbable.keanu.vertices.Vertex}'s state includes a boolean isObserved
 */
public interface Variable<VALUE, STATE extends VariableState> {

    VariableReference getReference();

    VALUE getValue();

    long[] getShape();

    STATE getState();
}
