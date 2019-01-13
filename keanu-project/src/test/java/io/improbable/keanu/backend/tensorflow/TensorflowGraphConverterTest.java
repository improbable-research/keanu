package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.ProbabilisticGraphWithGradient;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.LogProbGradientCalculator;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;
import org.junit.Ignore;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static com.google.common.collect.ImmutableMap.of;
import static org.junit.Assert.assertEquals;

public class TensorflowGraphConverterTest {

    @Test
    public void canRunSimpleAddition() {
        DoubleVertex A = new GaussianVertex(0, 1);
        A.setValue(2);
        DoubleVertex B = new GaussianVertex(1, 1);
        B.setValue(3);

        DoubleVertex C = A.plus(B);

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        Map<VariableReference, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        DoubleTensor result = graph.compute(inputs, C.getReference());

        assertEquals(C.getValue(), result);
    }

    @Test
    public void canRunDoubleTensorAddition() {
        DoubleVertex A = new GaussianVertex(new long[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new long[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex C = A.plus(B);

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        Map<VariableReference, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        DoubleTensor result = graph.compute(inputs, C.getReference());

        assertEquals(C.getValue(), result);
    }

    @Test
    public void canMaintainStateInGraph() {
        DoubleVertex A = new GaussianVertex(new long[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new long[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex C = A.plus(B);

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        DoubleTensor result = graph.compute(of(
            A.getReference(), A.getValue(),
            B.getReference(), B.getValue()
        ), C.getReference());

        assertEquals(C.eval(), result);

        DoubleTensor nextBValue = DoubleTensor.create(new double[]{0.1, 0.1, 0.1, 0.1}, 2, 2);
        DoubleTensor resultAfterRun = graph.compute(of(
            B.getReference(), nextBValue
        ), C.getReference());

        B.setValue(nextBValue);

        assertEquals(C.eval(), resultAfterRun);
    }

    @Test
    public void canRunIntegerTensorAddition() {
        IntegerVertex A = new UniformIntVertex(new long[]{2, 2}, 0, 1);
        A.setValue(IntegerTensor.create(new int[]{1, 2, 3, 4}, 2, 2));

        IntegerVertex B = new UniformIntVertex(new long[]{2, 2}, 1, 1);
        B.setValue(IntegerTensor.create(new int[]{5, 6, 7, 8}, 2, 2));

        IntegerVertex C = A.times(B).abs();

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        Map<VariableReference, IntegerTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        IntegerTensor result = graph.compute(inputs, C.getReference());

        assertEquals(C.getValue(), result);
    }

    @Test
    public void canRunTensorAnd() {
        BooleanVertex A = new BernoulliVertex(new long[]{2, 2}, 0.5);
        A.setValue(BooleanTensor.create(new boolean[]{true, false, true, false}, 2, 2));

        BooleanVertex B = new BernoulliVertex(new long[]{2, 2}, 0.75);
        B.setValue(BooleanTensor.create(new boolean[]{false, false, true, true}, 2, 2));

        BooleanVertex C = A.and(B).not();

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        Map<VariableReference, BooleanTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        BooleanTensor result = graph.compute(inputs, C.getReference());

        assertEquals(C.getValue(), result);
    }

    @Test
    public void canTensorConcat() {
        DoubleVertex A = new GaussianVertex(new long[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new long[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex C = DoubleVertex.concat(0, A, B);

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        Map<VariableReference, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        DoubleTensor result = graph.compute(inputs, C.getReference());

        assertEquals(C.getValue(), result);
    }

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
    public void canRunTensorMultiplication() {
        DoubleVertex A = new GaussianVertex(new long[]{2, 2}, 0, 1);
        A.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        DoubleVertex B = new GaussianVertex(new long[]{2, 2}, 1, 1);
        B.setValue(DoubleTensor.create(new double[]{5, 6, 7, 8}, 2, 2));

        DoubleVertex C = A.plus(B);
        DoubleVertex D = C.times(B);
        DoubleVertex out = D.matrixMultiply(C);

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        Map<VariableReference, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        DoubleTensor result = graph.compute(inputs, out.getReference());

        assertEquals(out.getValue(), result);
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
