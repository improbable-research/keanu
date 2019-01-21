package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.experimental.UtilityClass;

@UtilityClass
public class KeanuSimulatedAnnealing {

    public static SimulatedAnnealing withDefaultConfigFor(KeanuProbabilisticModel model) {
        return withDefaultConfigFor(model, KeanuRandom.getDefaultRandom());
    }

    public static SimulatedAnnealing withDefaultConfigFor(KeanuProbabilisticModel model, KeanuRandom random) {
        return SimulatedAnnealing.builder()
            .proposalDistribution(new PriorProposalDistribution(model.getLatentVertices()))
            .random(random)
            .build();
    }
}
