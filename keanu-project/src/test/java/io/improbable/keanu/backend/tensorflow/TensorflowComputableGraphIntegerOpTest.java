package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.UniformIntVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TensorflowComputableGraphIntegerOpTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

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
    public void canRunSum() {
        testIntegerUnaryOperation(new long[]{2, 3}, v -> v.sum(0));
        testIntegerUnaryOperation(new long[]{2, 3}, v -> v.sum(1));
        testIntegerUnaryOperation(new long[]{2, 3}, IntegerVertex::sum);
    }

    @Test
    public void canRunAbs() {
        testIntegerUnaryOperation(IntegerVertex::abs);
    }

    private void testIntegerUnaryOperation(Function<IntegerVertex, IntegerVertex> op) {
        testIntegerUnaryOperation(new long[0], op);
        testIntegerUnaryOperation(new long[]{2}, op);
        testIntegerUnaryOperation(new long[]{2, 2}, op);
    }

    private void testIntegerUnaryOperation(long[] shape, Function<IntegerVertex, IntegerVertex> op) {
        UniformIntVertex A = new UniformIntVertex(shape, 0, 1);
        A.setValue(A.sample());

        IntegerVertex out = op.apply(A);

        ComputableGraph graph = TensorflowComputableGraph.convert(out.getConnectedGraph());

        Map<VariableReference, IntegerTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());

        IntegerTensor result = graph.compute(inputs, out.getReference());

        assertEquals(out.getValue(), result);
    }
}
