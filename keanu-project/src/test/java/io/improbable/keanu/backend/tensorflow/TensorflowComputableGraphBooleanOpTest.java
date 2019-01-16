package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.DeterministicRule;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import org.junit.Rule;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static org.junit.Assert.assertEquals;

public class TensorflowComputableGraphBooleanOpTest {

    @Rule
    public DeterministicRule deterministicRule = new DeterministicRule();

    @Test
    public void canRunElementwiseAnd() {
        testBooleanBinaryOperation(BooleanVertex::and);
    }

    @Test
    public void canRunElementwiseOr() {
        testBooleanBinaryOperation(BooleanVertex::or);
    }

    @Test
    public void canRunElementwiseNot() {
        testBooleanUnaryOperation(v -> v.not());
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

    private void testBooleanUnaryOperation(Function<BooleanVertex, BooleanVertex> op) {
        testBooleanUnaryOperation(new long[0], op);
        testBooleanUnaryOperation(new long[]{2}, op);
        testBooleanUnaryOperation(new long[]{2, 2}, op);
    }

    private void testBooleanUnaryOperation(long[] shape, Function<BooleanVertex, BooleanVertex> op) {
        BernoulliVertex A = new BernoulliVertex(shape, 0.5);
        A.setValue(A.sample());

        BooleanVertex out = op.apply(A);

        ComputableGraph graph = TensorflowComputableGraph.convert(out.getConnectedGraph());

        Map<VariableReference, BooleanTensor> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());

        BooleanTensor result = graph.compute(inputs, out.getReference());

        assertEquals(out.getValue(), result);
    }
}
