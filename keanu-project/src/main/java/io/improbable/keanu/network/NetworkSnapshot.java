package io.improbable.keanu.network;


import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Saves the state (value and observed) of a specified collection of vertices.
 */
public class NetworkSnapshot {

    public static NetworkSnapshot create(Set<Variable> vertices) {
        return new NetworkSnapshot(vertices);
    }

    private final Map<Variable, Object> values;
    private final Set<Variable> observed;

    private NetworkSnapshot(Collection<Variable> vertices) {
        values = new HashMap<>();
        observed = new HashSet<>();
        for (Variable v : vertices) {
            values.put(v, v.getValue());
            if (v.isObserved()) {
                observed.add(v);
            }
        }
    }

    /**
     * Revert the state of the network to the previously saved values
     */
    public void apply() {
        for (Variable v : values.keySet()) {
            if (observed.contains(v)) {
                v.observe(values.get(v));
            } else {
                v.unobserve();
                v.setValue(values.get(v));
            }

        }
    }

}
