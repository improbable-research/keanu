package io.improbable.keanu.backend.tensorflow;

import static org.junit.Assert.assertEquals;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import org.junit.Test;

import io.improbable.keanu.backend.ProbabilisticGraph;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradient;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;

public class TensorflowGraphConverterTest {

    @Test
    public void canRunSimpleAddition() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(2);
        DoubleVertex B = new GaussianVertex(1, 1);
        B.setValue(3);

        A.setLabel(new VertexLabel("A"));
        B.setLabel(new VertexLabel("B"));

        DoubleVertex C = A.plus(B);

        String outputName = "someOutput";
        C.setLabel(new VertexLabel(outputName));

        ProbabilisticGraph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));

        Map<String, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getLabel().toString(), A.getValue());
        inputs.put(B.getLabel().toString(), B.getValue());

        DoubleTensor result = graph.getOutputs(inputs, Collections.singletonList(outputName)).get(0);

        assertEquals(C.getValue(), result);
    }

    @Test
    public void canRunTensorAddition() {
        DoubleVertex A = new GaussianVertex(new int[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new int[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        A.setLabel(new VertexLabel("A"));
        B.setLabel(new VertexLabel("B"));

        DoubleVertex C = A.plus(B);

        String outputName = "someOutput";
        C.setLabel(new VertexLabel(outputName));

        ProbabilisticGraph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));

        Map<String, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getLabel().toString(), A.getValue());
        inputs.put(B.getLabel().toString(), B.getValue());

        DoubleTensor result = graph.getOutputs(inputs, Collections.singletonList(outputName)).get(0);

        assertEquals(C.getValue(), result);
    }

    @Test
    public void canRunTensorMultiplication() {
        DoubleVertex A = new GaussianVertex(new int[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new int[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        A.setLabel(new VertexLabel("A"));
        B.setLabel(new VertexLabel("B"));

        DoubleVertex C = A.plus(B);
        DoubleVertex D = C.times(B);
        DoubleVertex out = D.matrixMultiply(C);

        String outputName = "output";
        out.setLabel(new VertexLabel(outputName));

        ProbabilisticGraph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));

        Map<String, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getLabel().toString(), A.getValue());
        inputs.put(B.getLabel().toString(), B.getValue());

        DoubleTensor result = graph.getOutputs(inputs, Collections.singletonList(outputName)).get(0);

        assertEquals(out.getValue(), result);
    }

    @Test
    public void canRunLogProbabilityAndGradientLogProbabilityOfGaussian() {

        long n = 20;
        int[] shape = new int[]{(int) n, (int) n};

        GaussianVertex A = new GaussianVertex(shape, 0, 1);
        GaussianVertex B = new GaussianVertex(shape, 1, 1);

        DoubleTensor initialA = A.sample();
        DoubleTensor initialB = B.sample();

        A.setValue(initialA);
        B.setValue(initialB);

        A.setLabel(new VertexLabel("A"));
        B.setLabel(new VertexLabel("B"));

        DoubleVertex C = A.matrixMultiply(B).matrixMultiply(A).times(0.5).matrixMultiply(B);

        DoubleVertex CObserved = new GaussianVertex(shape, C, 2);
        CObserved.observe(initialA);
        BayesianNetwork network = new BayesianNetwork(C.getConnectedGraph());

        double expectedLogProb = network.getLogOfMasterP();

        Map<VertexId, DoubleTensor> keanuGradients = LogProbGradient
            .getJointLogProbGradientWrtLatents(Probabilistic.keepOnlyProbabilisticVertices(network.getLatentOrObservedVertices()));

        try (ProbabilisticGraph graph = TensorflowGraphConverter.convert(network)) {

            Map<String, DoubleTensor> inputs = new HashMap<>();
            inputs.put(A.getLabel().toString(), initialA);
            inputs.put(B.getLabel().toString(), initialB);

            double tensorflowResult = graph.logProb(inputs);
            Map<String, DoubleTensor> tensorflowGradients = graph.logProbGradients(inputs);

            assertEquals(keanuGradients.get(A.getId()), tensorflowGradients.get("A"));
            assertEquals(keanuGradients.get(B.getId()), tensorflowGradients.get("B"));
            assertEquals(expectedLogProb, tensorflowResult, 1e-2);
        }

    }

}
