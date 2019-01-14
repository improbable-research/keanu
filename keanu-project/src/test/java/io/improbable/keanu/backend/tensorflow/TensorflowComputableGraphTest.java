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
import java.util.function.BiFunction;

import static com.google.common.collect.ImmutableMap.of;
import static org.junit.Assert.assertEquals;

public class TensorflowComputableGraphTest {

    @Test
    public void canRunElementwiseAddition() {
        testDoubleBinaryOperation(DoubleVertex::plus);
    }

    @Test
    public void canRunElementwiseSubtraction() {
        testDoubleBinaryOperation(DoubleVertex::minus);
    }

    @Test
    public void canRunElementwiseMultiplication() {
        testDoubleBinaryOperation(DoubleVertex::times);
    }

    @Test
    public void canRunElementwiseDivision() {
        testDoubleBinaryOperation(DoubleVertex::div);
    }

    @Test
    public void canRunMatrixMultiplication() {
        testDoubleBinaryOperation(new long[]{2, 2}, DoubleVertex::matrixMultiply);
    }

    private void testDoubleBinaryOperation(BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        testDoubleBinaryOperation(new long[0], op);
        testDoubleBinaryOperation(new long[]{2, 2}, op);
        testDoubleBinaryOperation(new long[]{2, 2, 2}, op);
    }

    private void testDoubleBinaryOperation(long[] shape, BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        DoubleVertex A = new GaussianVertex(shape, 0, 1);
        A.setValue(A.sample());

        DoubleVertex B = new GaussianVertex(shape, 1, 1);
        B.setValue(B.sample());

        DoubleVertex out = op.apply(A, B);

        ComputableGraph graph = TensorflowComputableGraph.convert(out.getConnectedGraph());

        Map<VariableReference, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        DoubleTensor result = graph.compute(inputs, out.getReference());

        assertEquals(out.getValue(), result);
    }

    @Test
    public void canRunIntegerElementwiseAddition() {
        testIntegerBinaryOperation(IntegerVertex::plus);
    }

    @Test
    public void canRunIntegerElementwiseSubtraction() {
        testIntegerBinaryOperation(IntegerVertex::minus);
    }

    @Test
    public void canRunIntegerElementwiseMultiplication() {
        testIntegerBinaryOperation(IntegerVertex::times);
    }

    @Test
    public void canRunIntegerElementwiseDivision() {
        testIntegerBinaryOperation(IntegerVertex::div);
    }

    private void testIntegerBinaryOperation(BiFunction<IntegerVertex, IntegerVertex, IntegerVertex> op) {
        testIntegerBinaryOperation(new long[0], op);
        testIntegerBinaryOperation(new long[]{2, 2}, op);
        testIntegerBinaryOperation(new long[]{2, 2, 2}, op);
    }

    private void testIntegerBinaryOperation(long[] shape, BiFunction<IntegerVertex, IntegerVertex, IntegerVertex> op) {
        IntegerVertex A = new UniformIntVertex(shape, 0, 1);
        A.setValue(A.sample());

        IntegerVertex B = new UniformIntVertex(shape, 1, 1);
        B.setValue(B.sample());

        IntegerVertex C = op.apply(A, B);

        ComputableGraph graph = TensorflowComputableGraph.convert(C.getConnectedGraph());

        Map<VariableReference, IntegerTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        IntegerTensor result = graph.compute(inputs, C.getReference());

        assertEquals(C.getValue(), result);
    }

    @Test
    public void canRunElementwiseAnd() {
        testBooleanBinaryOperation(BooleanVertex::and);
    }

    @Test
    public void canRunElementwiseOr() {
        testBooleanBinaryOperation(BooleanVertex::or);
    }

    private void testBooleanBinaryOperation(BiFunction<BooleanVertex, BooleanVertex, BooleanVertex> op) {
        testBooleanBinaryOperation(new long[0], op);
        testBooleanBinaryOperation(new long[]{2, 2}, op);
        testBooleanBinaryOperation(new long[]{2, 2, 2}, op);
    }

    private void testBooleanBinaryOperation(long[] shape, BiFunction<BooleanVertex, BooleanVertex, BooleanVertex> op) {
        BernoulliVertex A = new BernoulliVertex(shape, 0.5);
        A.setValue(A.sample());

        BernoulliVertex B = new BernoulliVertex(shape, 0.5);
        B.setValue(B.sample());

        BooleanVertex C = op.apply(A, B);

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

}
