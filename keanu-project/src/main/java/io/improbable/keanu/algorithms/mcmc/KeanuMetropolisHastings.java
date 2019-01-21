package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.experimental.UtilityClass;

import java.util.List;

@UtilityClass
public class KeanuMetropolisHastings {

    public static MetropolisHastings withDefaultConfigFor(KeanuProbabilisticModel model) {
        return withDefaultConfigFor(model, KeanuRandom.getDefaultRandom());
    }

    public static MetropolisHastings withDefaultConfigFor(KeanuProbabilisticModel model, KeanuRandom random) {
        List<Vertex> latentVertices = model.getLatentVertices();
        return MetropolisHastings.builder()
            .proposalDistribution(new PriorProposalDistribution(latentVertices))
            .logProbCalculationStrategy(new LambdaSectionOptimizedLogProbCalculator(latentVertices))
            .proposalApplicationStrategy(new CascadeOnApplication())
            .rejectionStrategy(new RollBackOnRejection(latentVertices))
            .random(random)
            .build();
    }
}
