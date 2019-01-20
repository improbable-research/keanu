package io.improbable.keanu.backend.keanu.compiled;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;

public class KeanuCompiledGraphTest {

    @Test
    public void compilesEmptyGraph() {

        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, ?> result = computableGraph.compute(Collections.emptyMap(), Collections.emptyList());

        assertTrue(result.isEmpty());
    }

    @Test
    public void compilesAddition() {
        assertBinaryMatches(DoubleVertex::plus);
    }

    @Test
    public void compilesSubtraction() {
        assertBinaryMatches(DoubleVertex::minus);
    }

    @Test
    public void compilesMultiplication() {
        assertBinaryMatches(DoubleVertex::times);
    }

    @Test
    public void compilesDivision() {
        assertBinaryMatches(DoubleVertex::div);
    }

    @Test
    public void compilesSeveralChainedOpsWithConstant() {
        assertBinaryMatches((a, b) -> a.plus(b).times(b).div(2).minus(a));
    }

    private void assertBinaryMatches(BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        assertBinaryMatches(new long[0], new long[0], op);
    }

    private void assertBinaryMatches(long[] shapeA, long[] shapeB, BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        GaussianVertex A = new GaussianVertex(shapeA, 0, 1);
        GaussianVertex B = new GaussianVertex(shapeB, 0, 1);

        DoubleVertex C = op.apply(A, B);

        compiler.convert(C.getConnectedGraph(), ImmutableList.of(C));

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        Map<VariableReference, ?> result = computableGraph.compute(inputs, Collections.emptyList());

        assertEquals(C.getValue(), result.get(C.getReference()));
    }

    @Test
    public void canAddDirectlyToGraph() {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        GaussianVertex A = new GaussianVertex( 0, 1);
        GaussianVertex B = new GaussianVertex( 0, 1);

        DoubleVertex C = A.times(B);

        compiler.convert(C.getConnectedGraph(), ImmutableList.of(C));

        VariableReference summation = compiler.add(A.getReference(), C.getReference());
        compiler.registerOutput(summation);

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        Map<VariableReference, ?> result = computableGraph.compute(inputs, Collections.emptyList());

        assertEquals(C.getValue(), result.get(C.getReference()));
        assertEquals(C.getValue().plus(A.getValue()), result.get(summation));
    }

    @Test
    public void compilesSum() {
        assertUnaryMatches(new long[]{2, 2}, DoubleVertex::sum);
        assertUnaryMatches(new long[]{2, 2}, (a) -> a.sum(0));
        assertUnaryMatches(new long[]{2, 2}, (a) -> a.sum(1));
    }

    @Test
    public void compilesLog() {
        assertUnaryMatches(DoubleVertex::abs);
        assertUnaryMatches(DoubleVertex::cos);
        assertUnaryMatches(DoubleVertex::exp);
        assertUnaryMatches(DoubleVertex::log);
        assertUnaryMatches(DoubleVertex::logGamma);
        assertUnaryMatches(DoubleVertex::sin);
        assertUnaryMatches(DoubleVertex::tan);
    }

    private void assertUnaryMatches(Function<DoubleVertex, DoubleVertex> op) {
        assertUnaryMatches(new long[0], op);
        assertUnaryMatches(new long[]{2}, op);
        assertUnaryMatches(new long[]{2, 2}, op);
    }

    private void assertUnaryMatches(long[] shape, Function<DoubleVertex, DoubleVertex> op) {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        UniformVertex A = new UniformVertex(shape, 0, 1);

        DoubleVertex C = op.apply(A);

        compiler.convert(C.getConnectedGraph(), ImmutableList.of(C));

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());

        Map<VariableReference, ?> result = computableGraph.compute(inputs, Collections.emptyList());

        assertEquals(C.getValue(), result.get(C.getReference()));
    }

}
