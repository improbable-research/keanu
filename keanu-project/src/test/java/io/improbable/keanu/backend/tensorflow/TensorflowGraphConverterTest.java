package io.improbable.keanu.backend.tensorflow;

import static org.junit.Assert.assertArrayEquals;

import java.nio.DoubleBuffer;
import java.util.Arrays;

import org.junit.Test;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

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

        DoubleVertex C = A.plus(B);

        String outputName = "someOutput";
        C.setLabel(new VertexLabel(outputName));

        Graph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));
        double[] result = runGraph(graph, outputName);

        assertArrayEquals(C.getValue().asFlatDoubleArray(), result, 1e-5);
    }

    @Test
    public void canRunTensorAddition() {
        DoubleVertex A = new GaussianVertex(new int[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new int[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex C = A.plus(B);

        String outputName = "someOutput";
        C.setLabel(new VertexLabel(outputName));

        Graph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));
        double[] result = runGraph(graph, outputName);

        assertArrayEquals(C.getValue().asFlatDoubleArray(), result, 1e-5);
    }

    @Test
    public void canRunTensorMultiplication() {
        DoubleVertex A = new GaussianVertex(new int[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new int[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex C = A.plus(B);
        DoubleVertex D = C.times(B);
        DoubleVertex out = D.matrixMultiply(C);

        String outputName = "output";
        out.setLabel(new VertexLabel(outputName));

        Graph graph = TensorflowGraphConverter.convert(new BayesianNetwork(C.getConnectedGraph()));
        double[] result = runGraph(graph, outputName);

        assertArrayEquals(out.getValue().asFlatDoubleArray(), result, 1e-5);
    }

    private double[] runGraph(Graph graph, String opName) {
        try (Session s = new Session(graph); Tensor result = s.runner().fetch(opName).run().get(0)) {

            DoubleBuffer buffer = DoubleBuffer.allocate(result.numElements());
            result.writeTo(buffer);
            double[] resultAsArray = buffer.array();

//            System.out.println(Arrays.toString(resultAsArray));

            return resultAsArray;
        }
    }

}
