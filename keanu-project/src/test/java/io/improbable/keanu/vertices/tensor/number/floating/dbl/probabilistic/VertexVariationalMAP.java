package io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;

import java.util.List;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorMatchers.valuesWithinEpsilonAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class VertexVariationalMAP {

    public static <T, VERTEX extends Vertex<T, VERTEX>> void inferHyperParamsFromSamples(
        Function<List<DoubleVertex>, Vertex<T, VERTEX>> vertexUnderTestCreator,
        List<DoubleVertex> hyperParamsForSampling,
        List<DoubleVertex> latentsToInfer,
        KeanuRandom random) {
        inferHyperParamsFromSamples(vertexUnderTestCreator, hyperParamsForSampling, latentsToInfer, 0.1, random);
    }

    public static <T, VERTEX extends Vertex<T, VERTEX>> void inferHyperParamsFromSamples(
        Function<List<DoubleVertex>, Vertex<T, VERTEX>> vertexUnderTestCreator,
        List<DoubleVertex> hyperParamsForSampling,
        List<DoubleVertex> latentsToInfer,
        Double epsilon,
        KeanuRandom random) {

        // SOURCE OF TRUTH
        Vertex sourceVertex = vertexUnderTestCreator.apply(hyperParamsForSampling);

        // GENERATE FAKE DATA
        T samples = ((Probabilistic<T>) sourceVertex).sample(random);

        Vertex<T, VERTEX> observedDistribution = vertexUnderTestCreator.apply(latentsToInfer);
        observedDistribution.observe(samples);

        // INFER HYPER PARAMETERS
        doInferenceOn(latentsToInfer.get(0), random);

        for (int i = 0; i < latentsToInfer.size(); i++) {
            assertThat(latentsToInfer.get(i).getValue(), valuesWithinEpsilonAndShapesMatch(hyperParamsForSampling.get(i).getValue(), epsilon));
        }
    }

    private static void doInferenceOn(DoubleVertex unknownVertex, KeanuRandom random) {
        BayesianNetwork inferNet = new BayesianNetwork(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroProbability(100, random);

        GradientOptimizer gradientOptimizer = Keanu.Optimizer.Gradient.of(inferNet);

        gradientOptimizer.maxAPosteriori();
    }

}
