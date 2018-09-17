package io.improbable.keanu.backend.tensorflow;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;

import com.google.common.primitives.Doubles;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
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
    public void canRunLogProbabilityOfGaussian() {

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

        DoubleTensor aGradient = keanuGradients.get(A.getId());
        DoubleTensor bGradient = keanuGradients.get(B.getId());

        int runCount = 1000;

        List<DoubleTensor> aInputs = new ArrayList<>();
        List<DoubleTensor> bInputs = new ArrayList<>();

        for (int i = 0; i < runCount; i++) {
            aInputs.add(A.sample());
            bInputs.add(B.sample());
        }

        List<Double> keanuResults = new ArrayList<>();
        long keanuRunStart = System.currentTimeMillis();
        for (int i = 0; i < runCount; i++) {
            A.setValue(aInputs.get(i));
            B.setValue(bInputs.get(i));
            VertexValuePropagation.cascadeUpdate(A, B);
            keanuResults.add(network.getLogOfMasterP());
        }

        long keanuRunTime = System.currentTimeMillis() - keanuRunStart;
        System.out.println("Keanu runtime: " + keanuRunTime + "ms");

        try (ProbabilisticGraph graph = TensorflowGraphConverter.convert(network)) {

            List<Double> tensorFlowResults = new ArrayList<>();
            Map<String, DoubleTensor> inputs = new HashMap<>();
            inputs.put(A.getLabel().toString(), initialA);
            inputs.put(B.getLabel().toString(), initialB);

            double result = graph.logProb(inputs);
            Map<String, DoubleTensor> gradients = graph.logProbGradients(inputs);
            assertEquals(aGradient, gradients.get("A"));
            assertEquals(bGradient, gradients.get("B"));

            long tensorFlowRunStart = System.currentTimeMillis();
            for (int i = 0; i < runCount; i++) {
                inputs.put(A.getLabel().toString(), aInputs.get(i));
                inputs.put(B.getLabel().toString(), bInputs.get(i));

                tensorFlowResults.add(graph.logProb(inputs));
            }

            long tensorFlowRunTime = System.currentTimeMillis() - tensorFlowRunStart;

            System.out.println("Tensorflow runtime: " + tensorFlowRunTime + "ms");
            System.out.println(String.format("%3.2f%%", (keanuRunTime / (double) tensorFlowRunTime) * 100));

            assertEquals(expectedLogProb, result, 1e-2);
            assertArrayEquals(Doubles.toArray(keanuResults), Doubles.toArray(tensorFlowResults), 1e-2);

        }

    }

}
