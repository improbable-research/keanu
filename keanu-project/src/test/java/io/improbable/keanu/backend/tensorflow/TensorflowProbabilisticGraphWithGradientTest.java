package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ProbabilisticGraphWithGradient;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Ignore;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class TensorflowProbabilisticGraphWithGradientTest {

    /**
     * Tensorflow does not support autodiff through a concat at the moment
     */
    @Test
    @Ignore
    public void canAutoDiffTensorConcat() {
        DoubleVertex A = new GaussianVertex(new long[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));
        A.setLabel("A");

        DoubleVertex B = new GaussianVertex(new long[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));
        B.setLabel("B");

        DoubleVertex C = new GaussianVertex(new long[]{2, 2}, 1, 1);
        C.setValue(DoubleTensor.create(new double[]{-3, 2, -4, 9}, 2, 2));
        C.setLabel("C");

        DoubleVertex D = DoubleVertex.concat(0, A.times(C), B.times(C));
        DoubleVertex E = new GaussianVertex(new long[]{4, 2}, D, 1);
        E.setLabel("E");
        E.observe(DoubleTensor.create(new double[]{0, 0, 0, 0, 0, 0, 0, 0}, 4, 2));

        Map<VariableReference, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());
        inputs.put(C.getReference(), C.getValue());
        inputs.put(E.getReference(), E.getValue());

        BayesianNetwork network = new BayesianNetwork(E.getConnectedGraph());
        double expectedLogProb = network.getLogOfMasterP();

        LogProbGradientCalculator calculator = new LogProbGradientCalculator(
            network.getLatentOrObservedVertices(),
            network.getContinuousLatentVertices()
        );

        Map<VertexId, DoubleTensor> keanuGradients = calculator.getJointLogProbGradientWrtLatents();

        try (ProbabilisticGraphWithGradient graph = TensorflowProbabilisticGraphWithGradient.convert(network)) {

            double tensorflowLogProb = graph.logProb(inputs);
            Map<? extends VariableReference, DoubleTensor> tensorflowGradients = graph.logProbGradients(inputs);

            assertEquals(keanuGradients.get(A.getId()), tensorflowGradients.get(A.getReference()));
            assertEquals(keanuGradients.get(B.getId()), tensorflowGradients.get(B.getReference()));
            assertEquals(keanuGradients.get(C.getId()), tensorflowGradients.get(C.getReference()));
            assertEquals(keanuGradients.get(E.getId()), tensorflowGradients.get(E.getReference()));
            assertEquals(expectedLogProb, tensorflowLogProb, 1e-2);
        }
    }

    @Test
    public void canRunLogProbabilityAndGradientLogProbabilityOfGaussian() {

        long n = 20;
        long[] shape = new long[]{n, n};

        GaussianVertex A = new GaussianVertex(shape, 0, 1);
        GaussianVertex B = new GaussianVertex(shape, 1, 1);

        DoubleTensor initialA = A.sample();
        DoubleTensor initialB = B.sample();

        A.setValue(initialA);
        B.setValue(initialB);

        DoubleVertex C = A.matrixMultiply(B).matrixMultiply(A).times(0.5).matrixMultiply(B);

        DoubleVertex CObserved = new GaussianVertex(shape, C, 2);
        CObserved.observe(initialA);
        BayesianNetwork network = new BayesianNetwork(C.getConnectedGraph());

        double expectedLogProb = network.getLogOfMasterP();

        LogProbGradientCalculator calculator = new LogProbGradientCalculator(
            network.getLatentOrObservedVertices(),
            network.getContinuousLatentVertices()
        );

        Map<VertexId, DoubleTensor> keanuGradients = calculator.getJointLogProbGradientWrtLatents();

        try (ProbabilisticGraphWithGradient graph = TensorflowProbabilisticGraphWithGradient.convert(network)) {

            Map<VariableReference, DoubleTensor> inputs = new HashMap<>();
            inputs.put(A.getReference(), initialA);
            inputs.put(B.getReference(), initialB);

            double tensorflowResult = graph.logProb(inputs);
            Map<? extends VariableReference, DoubleTensor> tensorflowGradients = graph.logProbGradients(inputs);

            assertEquals(keanuGradients.get(A.getId()), tensorflowGradients.get(A.getReference()));
            assertEquals(keanuGradients.get(B.getId()), tensorflowGradients.get(B.getReference()));
            assertEquals(expectedLogProb, tensorflowResult, 1e-2);
        }

    }

}
