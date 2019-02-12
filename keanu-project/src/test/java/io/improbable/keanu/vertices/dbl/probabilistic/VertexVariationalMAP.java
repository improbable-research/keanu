package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.variational.optimizer.gradient.GradientOptimizer;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class VertexVariationalMAP {

    public static void inferHyperParamsFromSamples(
        Function<List<DoubleVertex>, DoubleVertex> vertexUnderTestCreator,
        List<DoubleVertex> hyperParamsForSampling,
        List<DoubleVertex> latentsToInfer,
        KeanuRandom random) {

        // SOURCE OF TRUTH
        DoubleVertex sourceVertex = vertexUnderTestCreator.apply(hyperParamsForSampling);

        // GENERATE FAKE DATA
        DoubleTensor samples = ((Probabilistic<DoubleTensor>) sourceVertex).sample(random);

        DoubleVertex observedDistribution = vertexUnderTestCreator.apply(latentsToInfer);
        observedDistribution.observe(samples);

        // INFER HYPER PARAMETERS
        doInferenceOn(latentsToInfer.get(0), random);

        for (int i = 0; i < latentsToInfer.size(); i++) {
            assertEquals(hyperParamsForSampling.get(i).getValue().scalar(), latentsToInfer.get(i).getValue().scalar(), 0.1);
        }
    }

    private static void doInferenceOn(DoubleVertex unknownVertex, KeanuRandom random) {
        BayesianNetwork inferNet = new BayesianNetwork(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroProbability(100, random);

        GradientOptimizer gradientOptimizer = Keanu.Optimizer.Gradient.of(inferNet);

        gradientOptimizer.maxAPosteriori();
    }

}
