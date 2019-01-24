package io.improbable.keanu;

import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.graphtraversal.DifferentiableChecker;
import io.improbable.keanu.algorithms.mcmc.CascadeOnApplication;
import io.improbable.keanu.algorithms.mcmc.LambdaSectionOptimizedLogProbCalculator;
import io.improbable.keanu.algorithms.mcmc.RollBackOnRejection;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.KeanuProbabilisticModel;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import lombok.experimental.UtilityClass;

import java.util.List;

@UtilityClass
public class Keanu {

    @UtilityClass
    public static class Sampling {

        @UtilityClass
        public static class MCMC {

            /**
             * @param model network for which to choose sampling algorithm.
             * @return recommended sampling algorithm for this network.
             */
            public PosteriorSamplingAlgorithm withDefaultConfigFor(KeanuProbabilisticModel model) {
                return withDefaultConfigFor(model, KeanuRandom.getDefaultRandom());
            }

            /**
             * @param model  network for which to choose sampling algorithm.
             * @param random the random number generator.
             * @return recommended sampling algorithm for this network.
             */
            public PosteriorSamplingAlgorithm withDefaultConfigFor(KeanuProbabilisticModel model, KeanuRandom random) {
                if (DifferentiableChecker.isDifferentiableWrtLatents(model.getLatentOrObservedVertices())) {
                    return Keanu.Sampling.NUTS.withDefaultConfig(random);
                } else {
                    return Keanu.Sampling.MetropolisHastings.withDefaultConfigFor(model, random);
                }
            }
        }

        @UtilityClass
        public static class MetropolisHastings {

            public static io.improbable.keanu.algorithms.mcmc.MetropolisHastings withDefaultConfigFor(KeanuProbabilisticModel model) {
                return withDefaultConfigFor(model, KeanuRandom.getDefaultRandom());
            }

            public static io.improbable.keanu.algorithms.mcmc.MetropolisHastings withDefaultConfigFor(KeanuProbabilisticModel model, KeanuRandom random) {
                List<Vertex> latentVertices = model.getLatentVertices();
                return builder()
                    .proposalDistribution(new PriorProposalDistribution(latentVertices))
                    .logProbCalculationStrategy(new LambdaSectionOptimizedLogProbCalculator(latentVertices))
                    .proposalApplicationStrategy(new CascadeOnApplication())
                    .rejectionStrategy(new RollBackOnRejection(latentVertices))
                    .random(random)
                    .build();
            }

            public static io.improbable.keanu.algorithms.mcmc.MetropolisHastings.MetropolisHastingsBuilder builder() {
                return io.improbable.keanu.algorithms.mcmc.MetropolisHastings.builder();
            }
        }

        @UtilityClass
        public static class NUTS {

            public static io.improbable.keanu.algorithms.mcmc.nuts.NUTS withDefaultConfig() {
                return withDefaultConfig(KeanuRandom.getDefaultRandom());
            }

            public static io.improbable.keanu.algorithms.mcmc.nuts.NUTS withDefaultConfig(KeanuRandom random) {
                return builder()
                    .random(random)
                    .build();
            }

            public static io.improbable.keanu.algorithms.mcmc.nuts.NUTS.NUTSBuilder builder() {
                return io.improbable.keanu.algorithms.mcmc.nuts.NUTS.builder();
            }
        }
    }
}
