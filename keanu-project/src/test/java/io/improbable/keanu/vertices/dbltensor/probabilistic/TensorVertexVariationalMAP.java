package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.network.BayesNetDoubleAsContinuous;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TensorVertexVariationalMAP {

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
            assertEquals(hyperParamsForSampling.get(i).getValue().scalar(), latentsToInfer.get(i).getValue().scalar(), 0.1);
        }
    }

    private static void doInferenceOn(DoubleVertex unknownVertex, KeanuRandom random) {
        BayesNetDoubleAsContinuous inferNet = new BayesNetDoubleAsContinuous(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroMasterP(100, random);

        GradientOptimizer gradientOptimizer = new GradientOptimizer(inferNet);

        gradientOptimizer.maxAPosteriori(5000);
    }

}
