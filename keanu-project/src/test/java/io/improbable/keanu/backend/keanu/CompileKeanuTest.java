package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import org.junit.Test;

import java.util.Collections;
import java.util.Map;

import static junit.framework.TestCase.assertTrue;

public class CompileKeanuTest {

    @Test
    public void compilesEmptyGraph() {

        KeanuCompiler compiler = new KeanuCompiler();

        ComputableGraph computableGraph = compiler.compile(Collections.emptyList());

        Map<VariableReference, ?> result = computableGraph.compute(Collections.emptyMap(), Collections.emptyList());

        assertTrue(result.isEmpty());
    }
}
