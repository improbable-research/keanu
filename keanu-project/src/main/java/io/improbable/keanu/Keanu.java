package io.improbable.keanu;

import io.improbable.keanu.algorithms.PosteriorSamplingAlgorithm;
import io.improbable.keanu.algorithms.graphtraversal.DifferentiableChecker;
import io.improbable.keanu.algorithms.mcmc.RollBackToCachedValuesOnRejection;
import io.improbable.keanu.algorithms.mcmc.proposal.PriorProposalDistribution;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.Collection;
import java.util.List;
import java.util.Set;

/**
 * The entry point for creating {@link PosteriorSamplingAlgorithm}s such as {@link Sampling.MetropolisHastings} and {@link Sampling.NUTS}
 */
@UtilityClass
public class Keanu {

    @UtilityClass
    public static class Sampling {

        @UtilityClass
        /**
         * Class for choosing the appropriate sampling algorithm given a network.
         * If the given network is differentiable, NUTS is proposed, otherwise Metropolis Hastings is chosen.
         *
         * Usage:
         * PosteriorSamplingAlgorithm samplingAlgorithm = Keanu.Sampling.MCMC.withDefaultConfig(yourModel);
         * samplingAlgorithm.getPosteriorSamples(...);
         */
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
                    return Keanu.Sampling.MetropolisHastings.withDefaultConfig(random);
                }
            }
        }

        @UtilityClass
        public static class MetropolisHastings {

            public static io.improbable.keanu.algorithms.mcmc.MetropolisHastings withDefaultConfig() {
                return withDefaultConfig(KeanuRandom.getDefaultRandom());
            }

            public static io.improbable.keanu.algorithms.mcmc.MetropolisHastings withDefaultConfig(KeanuRandom random) {
                return builder()
                    .proposalDistribution(new PriorProposalDistribution())
                    .rejectionStrategy(new RollBackToCachedValuesOnRejection())
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

        @UtilityClass
        public static class SimulatedAnnealing {

            public static io.improbable.keanu.algorithms.mcmc.SimulatedAnnealing withDefaultConfig() {
                return withDefaultConfig(KeanuRandom.getDefaultRandom());
            }

            public static io.improbable.keanu.algorithms.mcmc.SimulatedAnnealing withDefaultConfig(KeanuRandom random) {
                return builder()
                    .proposalDistribution(new PriorProposalDistribution())
                    .rejectionStrategy(new RollBackToCachedValuesOnRejection())
                    .random(random)
                    .build();
            }

            public static io.improbable.keanu.algorithms.mcmc.SimulatedAnnealing.SimulatedAnnealingBuilder builder() {
                return io.improbable.keanu.algorithms.mcmc.SimulatedAnnealing.builder();
            }
        }
    }

    @UtilityClass
    public static class Optimizer {

        /**
         * Creates a Bayesian network from the given vertices and uses this to
         * create an {@link io.improbable.keanu.algorithms.variational.optimizer.Optimizer}. This provides methods for optimizing the values of latent variables
         * of the Bayesian network to maximise probability.
         *
         * @param vertices The vertices to create a Bayesian network from.
         * @return an {@link io.improbable.keanu.algorithms.variational.optimizer.Optimizer}
         */
        public io.improbable.keanu.algorithms.variational.optimizer.Optimizer of(Collection<? extends Vertex> vertices) {
            return of(new BayesianNetwork(vertices));
        }

        /**
         * Creates an {@link io.improbable.keanu.algorithms.variational.optimizer.Optimizer} which provides methods for optimizing the values of latent variables
         * of the Bayesian network to maximise probability.
         *
         * @param network The Bayesian network to run optimization on.
         * @return an {@link io.improbable.keanu.algorithms.variational.optimizer.Optimizer}
         */
        public io.improbable.keanu.algorithms.variational.optimizer.Optimizer of(BayesianNetwork network) {
            if (DifferentiableChecker.isDifferentiableWrtLatents(network.getLatentOrObservedVertices())) {
                return Gradient.of(network);
            } else {
                return NonGradient.of(network);
            }
        }

        /**
         * Creates a Bayesian network from the graph connected to the given vertex and uses this to
         * create an {@link io.improbable.keanu.algorithms.variational.optimizer.Optimizer}. This provides methods for optimizing the values of latent variables
         * of the Bayesian network to maximise probability.
         *
         * @param vertexFromNetwork A vertex in the graph to create the Bayesian network from.
         * @return an {@link io.improbable.keanu.algorithms.variational.optimizer.Optimizer}
         */
        public io.improbable.keanu.algorithms.variational.optimizer.Optimizer ofConnectedGraph(Vertex<?> vertexFromNetwork) {
            return of(vertexFromNetwork.getConnectedGraph());
        }

        @UtilityClass
        public class NonGradient {

            /**
             * Creates a BOBYQA {@link NonGradientOptimizer} which provides methods for optimizing the values of latent variables
             * of the Bayesian network to maximise probability.
             *
             * @param bayesNet The Bayesian network to run optimization on.
             * @return a {@link NonGradientOptimizer}
             */
            public NonGradientOptimizer of(BayesianNetwork bayesNet) {
                bayesNet.cascadeObservations();
                return builderFor(bayesNet).build();
            }

            /**
             * Creates a Bayesian network from the given vertices and uses this to
             * create a BOBYQA {@link NonGradientOptimizer}. This provides methods for optimizing the
             * values of latent variables of the Bayesian network to maximise probability.
             *
             * @param vertices The vertices to create a Bayesian network from.
             * @return a {@link NonGradientOptimizer}
             */
            public NonGradientOptimizer of(Collection<? extends Vertex> vertices) {
                return of(new BayesianNetwork(vertices));
            }

            /**
             * Creates a Bayesian network from the graph connected to the given vertex and uses this to
             * create a BOBYQA {@link NonGradientOptimizer}. This provides methods for optimizing the
             * values of latent variables of the Bayesian network to maximise probability.
             *
             * @param vertexFromNetwork A vertex in the graph to create the Bayesian network from
             * @return a {@link NonGradientOptimizer}
             */
            public NonGradientOptimizer ofConnectedGraph(Vertex<?> vertexFromNetwork) {
                return of(vertexFromNetwork.getConnectedGraph());
            }

            public NonGradientOptimizer.NonGradientOptimizerBuilder builderFor(Collection<? extends Vertex> vertices) {
                return builderFor(new BayesianNetwork(vertices));
            }

            public NonGradientOptimizer.NonGradientOptimizerBuilder builderFor(BayesianNetwork network) {
                initializeNetworkForOptimization(network);
                return NonGradientOptimizer.builder().probabilisticModel(new KeanuProbabilisticModel(network));
            }

        }

        @UtilityClass
        public class Gradient {

            /**
             * Creates a {@link GradientOptimizer} which provides methods for optimizing the values of latent variables
             * of the Bayesian network to maximise probability.
             *
             * @param bayesNet The Bayesian network to run optimization on.
             * @return a {@link GradientOptimizer}
             */
            public GradientOptimizer of(BayesianNetwork bayesNet) {
                return builderFor(bayesNet).build();
            }

            /**
             * Creates a Bayesian network from the given vertices and uses this to
             * create a {@link GradientOptimizer}. This provides methods for optimizing the values of latent variables
             * of the Bayesian network to maximise probability.
             *
             * @param vertices The vertices to create a Bayesian network from.
             * @return a {@link GradientOptimizer}
             */
            public GradientOptimizer of(Collection<? extends Vertex> vertices) {
                return of(new BayesianNetwork(vertices));
            }


            /**
             * Creates a Bayesian network from the graph connected to the given vertex and uses this to
             * create a {@link GradientOptimizer}. This provides methods for optimizing the values of latent variables
             * of the Bayesian network to maximise probability.
             *
             * @param vertexFromNetwork A vertex in the graph to create the Bayesian network from
             * @return a {@link GradientOptimizer}
             */
            public GradientOptimizer ofConnectedGraph(Vertex<?> vertexFromNetwork) {
                return of(vertexFromNetwork.getConnectedGraph());
            }

            public GradientOptimizer.GradientOptimizerBuilder builderFor(Set<Vertex> connectedGraph) {
                return builderFor(new BayesianNetwork(connectedGraph));
            }

            public GradientOptimizer.GradientOptimizerBuilder builderFor(BayesianNetwork network) {
                initializeNetworkForOptimization(network);
                return GradientOptimizer.builder().probabilisticModel(new KeanuProbabilisticModelWithGradient(network));
            }
        }

        void initializeNetworkForOptimization(BayesianNetwork bayesianNetwork) {
            List<Vertex> discreteLatentVertices = bayesianNetwork.getDiscreteLatentVertices();
            boolean containsDiscreteLatents = !discreteLatentVertices.isEmpty();

            if (containsDiscreteLatents) {
                throw new UnsupportedOperationException(
                    "Optimization unsupported on networks containing discrete latents. " +
                        "Found " + discreteLatentVertices.size() + " discrete latents.");
            }

            bayesianNetwork.cascadeObservations();
        }
    }
}
