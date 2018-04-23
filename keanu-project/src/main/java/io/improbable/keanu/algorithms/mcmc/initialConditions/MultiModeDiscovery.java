package io.improbable.keanu.algorithms.mcmc.initialConditions;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.SimulatedAnnealing;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.List;

public class MultiModeDiscovery {

    public static List<NetworkState> findModesBySimulatedAnnealing(BayesNet network, int attempts, int samplesPerAttempt) {

        List<NetworkState> maxSamples = new ArrayList<>();
        VertexValuePropagation.cascadeUpdate(network.getObservedVertices());
        List<? extends Vertex<?>> sortedByDependency = TopologicalSort.sort(network.getLatentVertices());

        for (int i = 0; i < attempts; i++) {
            BayesNet.setFromSampleAndCascade(sortedByDependency);
            NetworkState maxAPosteriori = SimulatedAnnealing.getMaxAPosteriori(network, samplesPerAttempt);
            maxSamples.add(maxAPosteriori);
        }

        return maxSamples;
    }
}
