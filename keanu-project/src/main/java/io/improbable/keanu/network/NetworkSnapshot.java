package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * Saves the state (value and observed) of a specified collection of variables.
 */
public class NetworkSnapshot {

    public static NetworkSnapshot create(Set<? extends Variable> variables) {
        return new NetworkSnapshot(variables);
    }

    private final Map<Variable, VariableState> variableStates;

    private NetworkSnapshot(Collection<? extends Variable> variables) {
        variableStates = new HashMap<>();
        for (Variable v : variables) {
            variableStates.put(v, v.getState());
        }
    }

    /**
     * Revert the state of the network to the previously saved state
     */
    public void apply() {
        for (Variable v : variableStates.keySet()) {
            v.setState(variableStates.get(v));
        }
    }

}
