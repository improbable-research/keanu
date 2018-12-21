package io.improbable.keanu.backend.keanu;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.vertices.Vertex;
import org.joor.Reflect;

import java.util.Collection;
import java.util.Map;
import java.util.function.BiFunction;

public class KeanuCompiler {

    public ComputableGraph compile(Collection<? extends Vertex> graph) {

        String source = getSource(graph);

        BiFunction<Map<VariableReference, ?>, Collection<VariableReference>, Map<VariableReference, ?>> computeFunction = Reflect.compile(
            "io.improbable.keanu.backend.keanu.CompiledKeanuGraph",
            source
        ).create().get();

        return new ComputableGraph() {

            @Override
            public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {
                return computeFunction.apply(inputs, outputs);
            }

            @Override
            public <T> T getInput(VariableReference input) {
                return null;
            }
        };
    }

    public String getSource(Collection<? extends Vertex> graph) {
        StringBuilder sourceString = new StringBuilder();

        startSource(sourceString);
        source(sourceString);
        endSource(sourceString);

        return sourceString.toString();
    }

    private void startSource(StringBuilder sb) {

        sb.append("package io.improbable.keanu.backend.keanu;\n");
        sb.append("import io.improbable.keanu.backend.VariableReference;\n");
        sb.append("import java.util.Collection;\n");
        sb.append("import java.util.Collections;\n");
        sb.append("import java.util.Map;\n");
        sb.append("public class CompiledKeanuGraph implements java.util.function.BiFunction<Map<VariableReference, ?>, Collection<VariableReference>, Map<VariableReference, ?>> {\n");
        sb.append("public Map<VariableReference, ?> apply(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {\n");
    }

    private void source(StringBuilder sb) {
        sb.append("return Collections.emptyMap();\n");
    }

    private void endSource(StringBuilder sb) {
        sb.append("}\n}\n");
    }
}
