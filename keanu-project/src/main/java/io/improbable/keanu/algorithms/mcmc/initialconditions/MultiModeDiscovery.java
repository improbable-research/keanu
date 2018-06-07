package io.improbable.keanu.algorithms.mcmc.initialconditions;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.mcmc.SimulatedAnnealing;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.ArrayList;
import java.util.List;

public class MultiModeDiscovery {

    private MultiModeDiscovery() {
    }

    public static List<NetworkState> findModesBySimulatedAnnealing(BayesianNetwork network,
                                                                   int attempts,
                                                                   int samplesPerAttempt,
                                                                   KeanuRandom random) {

        List<NetworkState> maxSamples = new ArrayList<>();
        network.cascadeObservations();
        List<Vertex> sortedByDependency = TopologicalSort.sort(network.getLatentVertices());

        for (int i = 0; i < attempts; i++) {
            BayesianNetwork.setFromSampleAndCascade(sortedByDependency, random);
            NetworkState maxAPosteriori = SimulatedAnnealing.getMaxAPosteriori(network, samplesPerAttempt);
            maxSamples.add(maxAPosteriori);
        }

        return maxSamples;
    }
}
