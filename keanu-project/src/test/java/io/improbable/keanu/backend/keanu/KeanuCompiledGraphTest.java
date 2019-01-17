package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;

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
        assertCompileMatches(DoubleVertex::plus);
    }

    @Test
    public void compilesSubtraction() {
        assertCompileMatches(DoubleVertex::minus);
    }

    @Test
    public void compilesMultiplication() {
        assertCompileMatches(DoubleVertex::times);
    }

    @Test
    public void compilesDivision() {
        assertCompileMatches(DoubleVertex::div);
    }

    @Test
    public void compilesSeveralChainedOps() {
        assertCompileMatches((a,b) -> a.plus(b).times(b).div(2).minus(a));
    }

    private void assertCompileMatches(BiFunction<DoubleVertex, DoubleVertex, DoubleVertex> op) {
        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);

        DoubleVertex C = op.apply(A, B);

        compiler.convert(C.getConnectedGraph());
        compiler.registerOutput(C.getReference());

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), A.getValue());
        inputs.put(B.getReference(), B.getValue());

        Map<VariableReference, ?> result = computableGraph.compute(inputs, Collections.emptyList());

        assertEquals(C.getValue(), result.get(C.getReference()));
    }
}
