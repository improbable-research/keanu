package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.algorithms.variational.TensorGradientOptimizer;
import io.improbable.keanu.network.BayesNetTensorAsContinuous;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;
import io.improbable.keanu.vertices.dbltensor.KeanuRandom;

import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TensorVertexVariationalMAP {

    public static void inferHyperParamsFromSamples(
        Function<List<DoubleTensorVertex>, DoubleTensorVertex> vertexUnderTestCreator,
        List<DoubleTensorVertex> hyperParamsForSampling,
        List<DoubleTensorVertex> latentsToInfer,
        KeanuRandom random) {

        // SOURCE OF TRUTH
        DoubleTensorVertex sourceVertex = vertexUnderTestCreator.apply(hyperParamsForSampling);

        // GENERATE FAKE DATA
        DoubleTensor samples = sourceVertex.sample(random);

        DoubleTensorVertex observedDistribution = vertexUnderTestCreator.apply(latentsToInfer);
        observedDistribution.observe(samples);

        // INFER HYPER PARAMETERS
        doInferenceOn(latentsToInfer.get(0), random);

        for (int i = 0; i < latentsToInfer.size(); i++) {
            assertEquals(hyperParamsForSampling.get(i).getValue().scalar(), latentsToInfer.get(i).getValue().scalar(), 0.1);
        }
    }

    private static void doInferenceOn(DoubleTensorVertex unknownVertex, KeanuRandom random) {
        BayesNetTensorAsContinuous inferNet = new BayesNetTensorAsContinuous(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroMasterP(100, random);

        TensorGradientOptimizer gradientOptimizer = new TensorGradientOptimizer(inferNet);

        gradientOptimizer.maxAPosteriori(5000);
    }

}
