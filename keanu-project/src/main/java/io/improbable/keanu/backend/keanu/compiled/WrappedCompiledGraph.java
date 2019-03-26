package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.backend.ComputableGraph;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static java.util.stream.Collectors.toMap;

/**
 * A computable graph that wraps a compiled Keanu graph. This compiled graph is generated at runtime from Keanu
 * vertices.
 */
class WrappedCompiledGraph implements ComputableGraph {

    private Map<String, VariableReference> outputsByString;
    private Function<Map<String, ?>, Map<String, ?>> computeFunction;

    private Map<VariableReference, Object> cachedInputs;

    WrappedCompiledGraph(Function<Map<String, ?>, Map<String, ?>> computeFunction,
                         List<VariableReference> outputs) {
        this.computeFunction = computeFunction;
        this.outputsByString = outputs.stream()
            .collect(toMap(VariableReference::toStringReference, output -> output));
        this.cachedInputs = new HashMap<>();
    }

    @Override
    public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs) {

        cachedInputs.putAll(inputs);

        final Map<String, Object> inputsByString = new HashMap<>();

        for (Map.Entry<VariableReference, ?> input : inputs.entrySet()) {
            inputsByString.put(input.getKey().toStringReference(), input.getValue());
        }

        final Map<String, ?> resultsByString = computeFunction.apply(inputsByString);

        final Map<VariableReference, Object> results = new HashMap<>();

        for (Map.Entry<String, ?> result : resultsByString.entrySet()) {
            results.put(outputsByString.get(result.getKey()), result.getValue());
        }

        return results;
    }

    @Override
    public <T> T getInput(VariableReference input) {
        return (T) cachedInputs.get(input);
    }
}
