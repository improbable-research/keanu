package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

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

        KeanuCompiledGraphBuilder compiler = new KeanuCompiledGraphBuilder();

        GaussianVertex A = new GaussianVertex(0, 1);
        GaussianVertex B = new GaussianVertex(0, 1);

        DoubleVertex C = A.plus(B);

        compiler.convert(C.getConnectedGraph());
        compiler.registerOutput(C.getReference());

        ComputableGraph computableGraph = compiler.build();

        Map<VariableReference, Object> inputs = new HashMap<>();
        inputs.put(A.getReference(), DoubleTensor.scalar(2));
        inputs.put(B.getReference(), DoubleTensor.scalar(3));

        Map<VariableReference, ?> result = computableGraph.compute(inputs, Collections.emptyList());

        assertEquals(5.0, ((DoubleTensor) result.get(C.getReference())).scalar());
    }
}
