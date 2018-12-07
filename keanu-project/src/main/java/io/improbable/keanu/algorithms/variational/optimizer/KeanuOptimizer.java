package io.improbable.keanu.algorithms.variational.optimizer;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.algorithms.variational.optimizer.nongradient.NonGradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.Collection;
import java.util.List;
import java.util.Set;

@UtilityClass
public class KeanuOptimizer {

    /**
     * Creates a Bayesian network from the given vertices and uses this to
     * create an {@link Optimizer}. This provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param vertices The vertices to create a Bayesian network from.
     * @return an {@link Optimizer}
     */
    public Optimizer of(Collection<? extends Vertex> vertices) {
        return of(new BayesianNetwork(vertices));
    }

    /**
     * Creates an {@link Optimizer} which provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param network The Bayesian network to run optimization on.
     * @return an {@link Optimizer}
     */
    public Optimizer of(BayesianNetwork network) {
        if (network.getDiscreteLatentVertices().isEmpty()) {
            return Gradient.of(network);
        } else {
            return NonGradient.of(network);
        }
    }

    /**
     * Creates a Bayesian network from the graph connected to the given vertex and uses this to
     * create an {@link Optimizer}. This provides methods for optimizing the values of latent variables
     * of the Bayesian network to maximise probability.
     *
     * @param vertexFromNetwork A vertex in the graph to create the Bayesian network from.
     * @return an {@link Optimizer}
     */
    public Optimizer ofConnectedGraph(Vertex<?> vertexFromNetwork) {
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
            return NonGradientOptimizer.builder().bayesianNetwork(new KeanuProbabilisticGraph(network));
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
            return GradientOptimizer.builder().bayesianNetwork(new KeanuProbabilisticWithGradientGraph(network));
        }
    }


    void initializeNetworkForOptimization(BayesianNetwork bayesianNetwork) {
        List<Vertex> discreteLatentVertices = bayesianNetwork.getDiscreteLatentVertices();
        boolean containsDiscreteLatents = !discreteLatentVertices.isEmpty();

        if (containsDiscreteLatents) {
            throw new UnsupportedOperationException(
                "Gradient optimization unsupported on networks containing discrete latents. " +
                    "Found " + discreteLatentVertices.size() + " discrete latents.");
        }

        bayesianNetwork.cascadeObservations();
    }
}
