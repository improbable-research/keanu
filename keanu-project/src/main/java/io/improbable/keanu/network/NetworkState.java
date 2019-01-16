package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.variational.optimizer.Variable;
import io.improbable.keanu.algorithms.variational.optimizer.VariableReference;

import java.util.Set;

/**
 * Depending on how a NetworkState is being used, it may be significantly more efficient
 * to use one underlying data structure vs another. The SimpleNetworkState is just a list
 * of values from each latent variable Map[Long, List] where as the List of NetworkStates
 * that the NetworkSamples class gives you is backed by a Map[Long, List[?]], which is
 * a more efficient data structure for sample collection.
 */
public interface NetworkState {

    <T> T get(Variable<T, ?> variable);

    <T> T get(VariableReference variableReference);

    Set<VariableReference> getVariableReferences();
}
