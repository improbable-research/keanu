package io.improbable.keanu.algorithms.mcmc;

import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.graphtraversal.DifferentiableChecker;
import io.improbable.keanu.algorithms.mcmc.nuts.NUTS;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

/**
 * Class for choosing the appropriate sampling algorithm given a network.
 * If the given network is differentiable, NUTS is proposed, otherwise Metropolis Hastings is chosen.
 *
 * Usage:
 * PosteriorSamplingAlgorithm samplingAlgorithm = MCMC.withDefaultConfig().forNetwork(yourNetwork);
 * samplingAlgorithm.getPosteriorSamples(...);
 */
@Builder
public class MCMC {

    @Getter
    @Setter
    @Builder.Default
    private KeanuRandom random = KeanuRandom.getDefaultRandom();

    public static MCMC withDefaultConfig() {
        return withDefaultConfig(KeanuRandom.getDefaultRandom());
    }

    public static MCMC withDefaultConfig(KeanuRandom random) {
        return MCMC.builder()
            .random(random)
            .build();
    }

    /**
     * @param bayesianNetwork network for which to choose sampling algorithm.
     * @return recommended sampling algorithm for this network.
     */
    public PosteriorSamplingAlgorithm forNetwork(BayesianNetwork bayesianNetwork) {
        if (DifferentiableChecker.isDifferentiable(bayesianNetwork.getLatentOrObservedVertices())) {
            return NUTS.withDefaultConfig(random);
        } else {
            return MetropolisHastings.withDefaultConfig(random);
        }
    }
}
