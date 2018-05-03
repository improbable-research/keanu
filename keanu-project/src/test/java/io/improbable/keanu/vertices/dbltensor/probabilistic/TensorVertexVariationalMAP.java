package io.improbable.keanu.vertices.dbltensor.probabilistic;

import io.improbable.keanu.algorithms.variational.GradientOptimizer;
import io.improbable.keanu.distributions.tensors.continuous.NDGaussian;
import io.improbable.keanu.network.BayesNet;
import io.improbable.keanu.plating.PlateBuilder;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;
import io.improbable.keanu.vertices.dbltensor.DoubleTensorVertex;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TensorVertexVariationalMAP {

    public static void inferHyperParamsFromSamples(
            Function<List<DoubleTensorVertex>, DoubleTensorVertex> vertexUnderTestCreator,
            List<DoubleTensorVertex> hyperParamsForSampling,
            List<DoubleTensorVertex> latentsToInfer,
            int numSamples) {

        // SOURCE OF TRUTH
        DoubleTensorVertex sourceVertex = vertexUnderTestCreator.apply(hyperParamsForSampling);

        // GENERATE FAKE DATA
        sourceVertex.setValue(DoubleTensor.zeros(new int[]{numSamples, 1}));
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
        BayesNet inferNet = new BayesNet(unknownVertex.getConnectedGraph());

        inferNet.probeForNonZeroMasterP(100);

        GradientOptimizer g = new GradientOptimizer(inferNet);

        g.maxAPosteriori(5000);
    }

}
