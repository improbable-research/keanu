package io.improbable.keanu.algorithms;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;
import io.improbable.keanu.network.NetworkState;
import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Map;
import java.util.Set;

/**
 * A network sample contains the state of the network (variable values) and the logOfMasterP at
 * a given point in time.
 */
@AllArgsConstructor
public class NetworkSample implements NetworkState {

    private final Map<VariableReference, ?> variableStates;

    @Getter
    private final double logOfMasterP;

    /**
     * @param variable the vertex to get the values of
     * @param <T>    the type of the values that the vertex contains
     * @return the values of the specified vertex
     */
    @Override
    public <T> T get(Variable<T, ?> variable) {
        return (T) variableStates.get(variable.getReference());
    }

    /**
     * @param variableReference the ID of the vertex to get the values of
     * @param <T>      the type of the values that the vertex contains
     * @return the values of the specified vertex
     */
    @Override
    public <T> T get(VariableReference variableReference) {
        return (T) variableStates.get(variableReference);
    }

    @Override
    public Set<VariableReference> getVariableReferences() {
        return variableStates.keySet();
    }
}
