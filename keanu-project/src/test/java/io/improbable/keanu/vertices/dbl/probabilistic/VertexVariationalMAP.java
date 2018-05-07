package io.improbable.keanu.vertices.dbl.probabilistic;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class VertexVariationalMAP {

    public static void inferHyperParamsFromSamples(
            Function<List<DoubleVertex>, DoubleVertex> vertexUnderTestCreator,
            List<DoubleVertex> hyperParamsForSampling,
            List<DoubleVertex> latentsToInfer,
            int numSamples) {

        // SOURCE OF TRUTH
        DoubleVertex sourceVertex = vertexUnderTestCreator.apply(hyperParamsForSampling);

        // GENERATE FAKE DATA
        List<Double> samples = getSamples(sourceVertex, numSamples);

        // OBSERVE
        new PlateBuilder<Double>()
                .fromIterator(samples.iterator())
                .withFactory((plate, sample) -> {

                    DoubleVertex observedDistribution = vertexUnderTestCreator.apply(latentsToInfer);

                    observedDistribution.observe(sample);

                }).build();

        // INFER HYPER PARAMETERS
        doInferenceOn(latentsToInfer.get(0));

        for (int i = 0; i < latentsToInfer.size(); i++) {
            assertEquals(hyperParamsForSampling.get(i).getValue(), latentsToInfer.get(i).getValue(), 0.1);
        }
    }

    private static void doInferenceOn(DoubleVertex unknownVertex) {
        BayesNet inferNet = new BayesNet(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroMasterP(100);

        GradientOptimizer g = new GradientOptimizer(inferNet);

        g.maxAPosteriori(5000);
    }

    private static List<Double> getSamples(DoubleVertex knownVertex, int numSamples) {

        List<Double> samples = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) {
            samples.add(knownVertex.sample());
        }

        return samples;
    }
}
