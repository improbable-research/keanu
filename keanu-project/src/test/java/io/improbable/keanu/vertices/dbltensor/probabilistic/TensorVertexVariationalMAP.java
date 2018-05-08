package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.algorithms.tensorVariational.TensorGradientOptimizer;
import io.improbable.keanu.network.TensorBayesNet;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;

import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TensorVertexVariationalMAP {

    public static void inferHyperParamsFromSamples(
            Function<List<DoubleTensorVertex>, DoubleTensorVertex> vertexUnderTestCreator,
            List<DoubleTensorVertex> hyperParamsForSampling,
            List<DoubleTensorVertex> latentsToInfer) {

        // SOURCE OF TRUTH
        DoubleTensorVertex sourceVertex = vertexUnderTestCreator.apply(hyperParamsForSampling);

        // GENERATE FAKE DATA
        DoubleTensor samples = sourceVertex.sample();

        DoubleTensorVertex observedDistribution = vertexUnderTestCreator.apply(latentsToInfer);
        observedDistribution.observe(samples);

        // INFER HYPER PARAMETERS
        doInferenceOn(latentsToInfer.get(0));

        for (int i = 0; i < latentsToInfer.size(); i++) {
            assertEquals(hyperParamsForSampling.get(i).getValue().scalar(), latentsToInfer.get(i).getValue().scalar(), 0.1);
        }
    }

    private static void doInferenceOn(DoubleTensorVertex unknownVertex) {
        TensorBayesNet inferNet = new TensorBayesNet(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroMasterP(100);

        TensorGradientOptimizer gradientOptimizer = new TensorGradientOptimizer(inferNet);

        gradientOptimizer.maxLikelihood(5000);
    }

}
