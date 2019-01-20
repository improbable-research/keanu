package io.improbable.keanu.backend.keanu.compiled;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static java.util.stream.Collectors.toMap;

class WrappedCompiledGraph implements ComputableGraph {

    private Map<String, VariableReference> outputsByString;
    private Function<Map<String, ?>, Map<String, ?>> computeFunction;

    WrappedCompiledGraph(Function<Map<String, ?>, Map<String, ?>> computeFunction,
                         Map<VariableReference, Object> constantValues,
                         List<VariableReference> outputs) {
        this.computeFunction = computeFunction;
        this.outputsByString = outputs.stream()
            .collect(toMap(VariableReference::toStringReference, output -> output));
    }

    @Override
    public Map<VariableReference, ?> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {

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
        return null;
    }
}
