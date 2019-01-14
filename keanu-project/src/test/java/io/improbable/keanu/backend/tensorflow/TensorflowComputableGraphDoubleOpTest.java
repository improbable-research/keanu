package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TensorflowComputableGraphDoubleOpTest {

    @Test
    public void canRunElementwiseAddition() {
        testDoubleBinaryOperationElementwise(DoubleVertex::plus);
    }

    @Test
    public void canRunBroadcastAddition() {
        testDoubleBinaryOperationBroadcast(DoubleVertex::plus);
    }

    @Test
    public void canRunElementwiseSubtraction() {
        testDoubleBinaryOperationElementwise(DoubleVertex::minus);
    }

    @Test
    public void canRunBroadcastSubtraction() {
        testDoubleBinaryOperationBroadcast(DoubleVertex::minus);
    }

    @Test
    public void canRunElementwiseMultiplication() {
        testDoubleBinaryOperationElementwise(DoubleVertex::times);
    }

    @Test
    public void canRunBroadcastMultiplication() {
        testDoubleBinaryOperationBroadcast(DoubleVertex::times);
    }

    @Test
    public void canRunElementwiseDivision() {
        testDoubleBinaryOperationElementwise(DoubleVertex::div);
    }

    @Test
    public void canRunBroadcastDivision() {
        testDoubleBinaryOperationBroadcast(DoubleVertex::div);
    }

    @Test
    public void canRunMatrixMultiplication() {
        testDoubleBinaryOperation(new long[]{2, 3}, new long[]{3, 2}, DoubleVertex::matrixMultiply);
    }

    @Test
    public void canRunConcat() {
        testDoubleBinaryOperation(new long[]{2, 3}, new long[]{1, 3}, (a, b) -> DoubleVertex.concat(0, a, b));
        testDoubleBinaryOperation(new long[]{2, 3}, new long[]{2, 4}, (a, b) -> DoubleVertex.concat(1, a, b));
    }

    private void testDoubleBinaryOperationElementwise(BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        testDoubleBinaryOperation(new long[0], new long[0], op);
        testDoubleBinaryOperation(new long[]{2}, new long[]{2}, op);
        testDoubleBinaryOperation(new long[]{2, 2}, new long[]{2, 2}, op);
        testDoubleBinaryOperation(new long[]{2, 2, 2}, new long[]{2, 2, 2}, op);
    }

    private void testDoubleBinaryOperationBroadcast(BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        testDoubleBinaryOperation(new long[0], new long[0], op);
        testDoubleBinaryOperation(new long[0], new long[]{2}, op);
        testDoubleBinaryOperation(new long[]{2, 2}, new long[]{1, 2}, op);
        testDoubleBinaryOperation(new long[]{2, 2, 2}, new long[]{1, 2, 2}, op);
    }

    private void testDoubleBinaryOperation(long[] leftShape, long[] rightShape, BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        DoubleVertex A = new UniformVertex(leftShape, 0.1, 1);
        A.setValue(A.sample());

        DoubleVertex B = new UniformVertex(rightShape, 0.1, 1);
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
    public void canRunAbs() {
        testDoubleUnaryOperation(DoubleVertex::abs);
    }

    @Test
    public void canRunExp() {
        testDoubleUnaryOperation(DoubleVertex::exp);
    }

    @Test
    public void canRunLog() {
        testDoubleUnaryOperation(DoubleVertex::log);
    }

    @Test
    public void canRunSum() {
        testDoubleUnaryOperation(new long[]{2, 3}, DoubleVertex::sum);
        testDoubleUnaryOperation(new long[]{2, 3}, v -> v.sum(1));
        testDoubleUnaryOperation(new long[]{2, 3}, v -> v.sum(0));
    }

    private void testDoubleUnaryOperation(Function<DoubleVertex, DoubleVertex> op) {
        testDoubleUnaryOperation(new long[0], op);
        testDoubleUnaryOperation(new long[]{2}, op);
        testDoubleUnaryOperation(new long[]{2, 2}, op);
    }

    private void testDoubleUnaryOperation(long[] shape, Function<DoubleVertex, DoubleVertex> op) {
        DoubleVertex A = new UniformVertex(shape, 0.1, 1);
        A.setValue(A.sample());

        DoubleVertex out = op.apply(A);

        ComputableGraph graph = TensorflowComputableGraph.convert(out.getConnectedGraph());

        Map<VariableReference, DoubleTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());

        DoubleTensor result = graph.compute(inputs, out.getReference());

        assertEquals(out.getValue(), result);
    }
}
