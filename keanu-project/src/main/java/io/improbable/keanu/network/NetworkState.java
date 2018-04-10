package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;

import java.util.Set;

/**
 * Depending on how a NetworkState is being used, it may be significantly more efficient
 * to use one underlying data structure vs another. The SimpleNetworkState is just a list
 * of values from each latent vertex Map[String, List] where as the List of NetworkStates
 * that the NetworkSamples class gives you is backed by a Map[String, List[?]], which is
 * a more efficient data structure for sample collection.
 */
public interface NetworkState {

    <T> T get(Vertex<T> vertex);

    <T> T get(String vertexId);

    Set<String> getVertexIds();
}
