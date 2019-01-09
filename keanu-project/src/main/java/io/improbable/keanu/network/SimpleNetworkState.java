package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;

import java.util.Map;
import java.util.Set;

public class SimpleNetworkState implements NetworkState {

    private final Map<VariableReference, ?> vertexValues;

    public SimpleNetworkState(Map<VariableReference, ?> vertexValues) {
        this.vertexValues = vertexValues;
    }

    @Override
    public <T> T get(Variable<T> vertex) {
        return (T) vertexValues.get(vertex.getReference());
    }

    @Override
    public <T> T get(VariableReference vertexId) {
        return (T) vertexValues.get(vertexId);
    }

    @Override
    public Set<VariableReference> getVertexIds() {
        return vertexValues.keySet();
    }
}
