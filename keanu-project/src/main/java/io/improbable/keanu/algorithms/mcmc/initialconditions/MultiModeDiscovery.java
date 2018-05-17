package io.improbable.keanu.algorithms.mcmc.initialconditions;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.mcmc.SimulatedAnnealing;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MultiModeDiscovery {

    private MultiModeDiscovery() {
    }

    public static List<NetworkState> findModesBySimulatedAnnealing(BayesNet network,
                                                                   int attempts,
                                                                   int samplesPerAttempt,
                                                                   KeanuRandom random) {

        List<NetworkState> maxSamples = new ArrayList<>();
        VertexValuePropagation.cascadeUpdate(network.getObservedVertices());
        List<Vertex> sortedByDependency = TopologicalSort.sort(network.getLatentVertices());

        for (int i = 0; i < attempts; i++) {
            BayesNet.setFromSampleAndCascade(sortedByDependency, random);
            NetworkState maxAPosteriori = SimulatedAnnealing.getMaxAPosteriori(network, samplesPerAttempt);
            maxSamples.add(maxAPosteriori);
        }

        return maxSamples;
    }
}
