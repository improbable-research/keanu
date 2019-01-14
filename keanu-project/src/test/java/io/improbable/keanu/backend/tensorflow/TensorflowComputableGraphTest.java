package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static com.google.common.collect.ImmutableMap.of;
import static org.junit.Assert.assertEquals;

public class TensorflowComputableGraphTest {

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
}
