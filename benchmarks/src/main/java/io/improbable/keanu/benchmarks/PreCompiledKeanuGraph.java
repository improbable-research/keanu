package io.improbable.keanu.benchmarks;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;

import static java.util.stream.Collectors.toMap;

public class PreCompiledKeanuGraph implements ComputableGraph {

    private final Map<String, VariableReference> outputsByString;
    private final Function<Map<String, ?>, Map<String, ?>> computeFunction;

    public PreCompiledKeanuGraph(VariableReference c) {

        computeFunction = new TestGraph();

        outputsByString = new HashMap<>();
        outputsByString.put(toSourceVariableName(c), c);
    }

    @Override
    public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {

        Map<String, Object> inputsByString = inputs.entrySet().stream()
            .collect(toMap(e -> toSourceVariableName(e.getKey()), Map.Entry::getValue));

        Map<String, ?> results = computeFunction.apply(inputsByString);

        return results.entrySet().stream()
            .collect(toMap(
                e -> outputsByString.get(e.getKey()),
                Map.Entry::getValue)
            );
    }

    @Override
    public <T> T getInput(VariableReference input) {
        throw new UnsupportedOperationException("No need to implement this for performance tests");
    }

    private String toSourceVariableName(VariableReference variableReference) {
        return variableReference.toStringReference();
    }

    public static class TestGraph implements java.util.function.Function<Map<String, ?>, Map<String, ?>> {

        public Map<String, ?> apply(Map<String, ?> inputs) {

            DoubleTensor A = (DoubleTensor) inputs.get("2");
            DoubleTensor B = (DoubleTensor) inputs.get("5");

            DoubleTensor C = (DoubleTensor) A.times(B);

            Map<String, Object> results = new HashMap<>();
            results.put("6", C);

            return results;
        }

    }
}
