package io.improbable.keanu.vertices.dbl.probabilistic;

import static org.junit.Assert.assertEquals;

import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import java.util.List;
import java.util.function.Function;

public class VertexVariationalMAP {

    public static void inferHyperParamsFromSamples(
            Function<List<DoubleVertex>, DoubleVertex> vertexUnderTestCreator,
            List<DoubleVertex> hyperParamsForSampling,
            List<DoubleVertex> latentsToInfer,
            KeanuRandom random) {

        // SOURCE OF TRUTH
        DoubleVertex sourceVertex = vertexUnderTestCreator.apply(hyperParamsForSampling);

        // GENERATE FAKE DATA
        DoubleTensor samples = sourceVertex.sample(random);

        DoubleVertex observedDistribution = vertexUnderTestCreator.apply(latentsToInfer);
        observedDistribution.observe(samples);

        // INFER HYPER PARAMETERS
        doInferenceOn(latentsToInfer.get(0), random);

        for (int i = 0; i < latentsToInfer.size(); i++) {
            assertEquals(
                    hyperParamsForSampling.get(i).getValue().scalar(),
                    latentsToInfer.get(i).getValue().scalar(),
                    0.1);
        }
    }

    private static void doInferenceOn(DoubleVertex unknownVertex, KeanuRandom random) {
        BayesianNetwork inferNet = new BayesianNetwork(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroProbability(100, random);

        GradientOptimizer gradientOptimizer = GradientOptimizer.of(inferNet);

        gradientOptimizer.maxAPosteriori();
    }
}
