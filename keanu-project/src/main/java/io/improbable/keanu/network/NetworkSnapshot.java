package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Saves the state (value and observed) of a specified collection of vertices.
 */
public class NetworkSnapshot {

    public static NetworkSnapshot create(Set<? extends Variable> vertices) {
        return new NetworkSnapshot(vertices);
    }

    private final Map<Variable, VariableState> values;

    private NetworkSnapshot(Collection<? extends Variable> vertices) {
        values = new HashMap<>();
        for (Variable v : vertices) {
            values.put(v, v.getState());
        }
    }

    /**
     * Revert the state of the network to the previously saved values
     */
    public void apply() {
        for (Variable v : values.keySet()) {
            v.setState(values.get(v));
        }
    }

}
