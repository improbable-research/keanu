package io.improbable.keanu.algorithms.mcmc.initialconditions;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.NetworkState;
import io.improbable.keanu.vertices.Vertex;

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
            KeanuProbabilisticModel model = new KeanuProbabilisticModel(network);
            NetworkState maxAPosteriori = Keanu.Sampling.SimulatedAnnealing.withDefaultConfig(random).getMaxAPosteriori(model, samplesPerAttempt);
            maxSamples.add(maxAPosteriori);
        }

        return maxSamples;
    }
}
