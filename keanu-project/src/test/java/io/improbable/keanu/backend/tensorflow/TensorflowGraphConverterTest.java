package io.improbable.keanu.backend.tensorflow;

import static org.junit.Assert.assertArrayEquals;

import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import com.google.common.primitives.Doubles;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
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

        Graph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));

        Map<String, Tensor> inputs = new HashMap<>();
        inputs.put(A.getLabel().toString(), Tensor.create(new long[]{1}, DoubleBuffer.wrap(A.getValue().asFlatDoubleArray())));
        inputs.put(B.getLabel().toString(), Tensor.create(new long[]{1}, DoubleBuffer.wrap(B.getValue().asFlatDoubleArray())));

        try (Session s = new Session(graph)) {
            double[] result = runGraph(s, outputName, inputs);
            assertArrayEquals(C.getValue().asFlatDoubleArray(), result, 1e-5);
        }
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

        Graph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));

        Map<String, Tensor> inputs = new HashMap<>();
        inputs.put(A.getLabel().toString(), Tensor.create(new long[]{2, 2}, DoubleBuffer.wrap(A.getValue().asFlatDoubleArray())));
        inputs.put(B.getLabel().toString(), Tensor.create(new long[]{2, 2}, DoubleBuffer.wrap(B.getValue().asFlatDoubleArray())));

        try (Session s = new Session(graph)) {
            double[] result = runGraph(s, outputName, inputs);
            assertArrayEquals(C.getValue().asFlatDoubleArray(), result, 1e-5);
        }

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

        Graph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));

        Map<String, Tensor> inputs = new HashMap<>();
        inputs.put(A.getLabel().toString(), Tensor.create(new long[]{2, 2}, DoubleBuffer.wrap(A.getValue().asFlatDoubleArray())));
        inputs.put(B.getLabel().toString(), Tensor.create(new long[]{2, 2}, DoubleBuffer.wrap(B.getValue().asFlatDoubleArray())));

        try (Session s = new Session(graph)) {
            double[] result = runGraph(s, outputName, inputs);
            assertArrayEquals(out.getValue().asFlatDoubleArray(), result, 1e-5);
        }
    }

    @Test
    public void canRunLogProbabilityOfGaussian() {

        long n = 20;
        int[] shape = new int[]{(int) n, (int) n};
        long[] shapeLong = new long[]{n, n};

        DoubleVertex A = new GaussianVertex(shape, 0, 1);
        DoubleVertex B = new GaussianVertex(shape, 1, 1);

        double[] initialA = A.sample().asFlatDoubleArray();
        double[] initialB = B.sample().asFlatDoubleArray();

        A.setValue(DoubleTensor.create(initialA, shape));
        B.setValue(DoubleTensor.create(initialB, shape));

        A.setLabel(new VertexLabel("A"));
        B.setLabel(new VertexLabel("B"));

        DoubleVertex C = A.matrixMultiply(B).matrixMultiply(A).times(0.5).matrixMultiply(B);

        DoubleVertex CObserved = new GaussianVertex(shape, C, 2);
        CObserved.observe(DoubleTensor.create(initialA, shape));
        BayesianNetwork network = new BayesianNetwork(C.getConnectedGraph());

        double[] expectedLogProb = new double[]{network.getLogOfMasterP()};

        int runCount = 1000;

        List<double[]> aInputs = new ArrayList<>();
        List<double[]> bInputs = new ArrayList<>();

        for (int i = 0; i < runCount; i++) {
            aInputs.add(A.sample().asFlatDoubleArray());
            bInputs.add(B.sample().asFlatDoubleArray());
        }

        List<Double> keanuResults = new ArrayList<>();
        long keanuRunStart = System.currentTimeMillis();
        for (int i = 0; i < runCount; i++) {
            A.setValue(DoubleTensor.create(aInputs.get(i), shape));
            B.setValue(DoubleTensor.create(bInputs.get(i), shape));
            VertexValuePropagation.cascadeUpdate(A, B);
            keanuResults.add(network.getLogOfMasterP());
        }

        long keanuRunTime = System.currentTimeMillis() - keanuRunStart;
        System.out.println("Keanu runtime: " + keanuRunTime + "ms");

        Graph graph = TensorflowGraphConverter.convert(network);

        List<Double> tensorFlowResults = new ArrayList<>();
        try (Session s = new Session(graph)) {

            Map<String, Tensor> inputs = new HashMap<>();
            inputs.put(A.getLabel().toString(), Tensor.create(shapeLong, DoubleBuffer.wrap(initialA)));
            inputs.put(B.getLabel().toString(), Tensor.create(shapeLong, DoubleBuffer.wrap(initialB)));

            double[] result = runGraph(s, TensorflowGraphConverter.LOG_PROB_LABEL, inputs);


            long tensorFlowRunStart = System.currentTimeMillis();
            for (int i = 0; i < runCount; i++) {
                inputs.put(A.getLabel().toString(), Tensor.create(shapeLong, DoubleBuffer.wrap(aInputs.get(i))));
                inputs.put(B.getLabel().toString(), Tensor.create(shapeLong, DoubleBuffer.wrap(bInputs.get(i))));

                tensorFlowResults.add(runGraph(s, TensorflowGraphConverter.LOG_PROB_LABEL, inputs)[0]);
            }

            long tensorFlowRunTime = System.currentTimeMillis() - tensorFlowRunStart;
            System.out.println("Tensorflow runtime: " + tensorFlowRunTime + "ms");

            System.out.println(String.format("%3.2f%%", (keanuRunTime / (double) tensorFlowRunTime) * 100));
            assertArrayEquals(expectedLogProb, result, 1e-2);
        }

        assertArrayEquals(Doubles.toArray(keanuResults), Doubles.toArray(tensorFlowResults), 1e-2);
    }

    private double[] runGraph(Session s, String outputName, Map<String, Tensor> inputs) {

        Session.Runner runner = s.runner();
        for (Map.Entry<String, Tensor> inputEntry : inputs.entrySet()) {
            runner = runner.feed(inputEntry.getKey(), inputEntry.getValue());
        }

        try (Tensor result = runner.fetch(outputName).run().get(0)) {

            DoubleBuffer buffer = DoubleBuffer.allocate(result.numElements());
            result.writeTo(buffer);
            double[] resultAsArray = buffer.array();

//            System.out.println(Arrays.toString(resultAsArray));

            return resultAsArray;
        }
    }

}
