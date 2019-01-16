package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;

import java.util.Map;
import java.util.Set;

public class SimpleNetworkState implements NetworkState {

    private final Map<VariableReference, ?> variableValues;

    public SimpleNetworkState(Map<VariableReference, ?> variableValues) {
        this.variableValues = variableValues;
    }

    @Override
    public <T> T get(Variable<T, ?> variable) {
        return (T) variableValues.get(variable.getReference());
    }

    @Override
    public <T> T get(VariableReference variableReference) {
        return (T) variableValues.get(variableReference);
    }

    @Override
    public Set<VariableReference> getVariableReferences() {
        return variableValues.keySet();
    }
}
