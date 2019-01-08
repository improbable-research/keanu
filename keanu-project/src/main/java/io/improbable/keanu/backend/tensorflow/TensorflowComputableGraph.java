package io.improbable.keanu.backend.tensorflow;

import io.improbable.keanu.backend.ComputableGraph;
import io.improbable.keanu.backend.VariableReference;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import lombok.Getter;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Scope;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static io.improbable.keanu.backend.tensorflow.TensorflowData.toBooleanTensor;
import static io.improbable.keanu.backend.tensorflow.TensorflowData.toDoubleTensor;
import static io.improbable.keanu.backend.tensorflow.TensorflowData.toIntegerTensor;
import static io.improbable.keanu.backend.tensorflow.TensorflowData.toTensorFlow;
import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.toList;
import static java.util.stream.Collectors.toMap;

public class TensorflowComputableGraph implements ComputableGraph {

    public static TensorflowComputableGraph convert(Set<Vertex> vertices) {
        TensorflowGraphBuilder graphBuilder = new TensorflowGraphBuilder();
        graphBuilder.convert(vertices);
        return graphBuilder.build();
    }

    private final Session session;

    @Getter
    private final Scope scope;

    private Map<VariableReference, Object> inputCache = new HashMap<>();

    public TensorflowComputableGraph(Session session, Scope scope, Map<VariableReference, Object> defaultInputs) {
        this.session = session;
        this.scope = scope;
        this.inputCache.putAll(defaultInputs);
    }

    public TensorflowComputableGraph(Session session, Scope scope) {
        this(session, scope, Collections.emptyMap());
    }

    @Override
    public <T> T compute(Map<VariableReference, ?> inputs, VariableReference output) {
        return (T) compute(inputs, singletonList(output)).get(output);
    }

    @Override
    public Map<VariableReference, Object> compute(Map<VariableReference, ?> inputs, Collection<VariableReference> outputs) {

        cacheInputs(inputs);

        List<VariableReference> outputsNotFed = filterInputs(inputCache, outputs);
        Session.Runner runner = feedInputs(inputCache);
        List<Tensor<?>> tfResults = fetchOutputs(runner, outputsNotFed).run();

        Map<VariableReference, Object> resultsAsKeanu = convertToKeanuTensors(outputsNotFed, tfResults);

        Map<VariableReference, Object> inputsFromOutputs = getOutputValuesThatAreInputs(inputCache, outputs);

        resultsAsKeanu.putAll(inputsFromOutputs);

        return resultsAsKeanu;
    }

    private Map<VariableReference, Object> getOutputValuesThatAreInputs(Map<VariableReference, ?> inputs,
                                                                        Collection<VariableReference> outputs) {
        return outputs.stream()
            .filter(inputs::containsKey)
            .collect(toMap(output -> output, inputs::get));
    }

    private List<VariableReference> filterInputs(Map<VariableReference, ?> inputs,
                                                 Collection<VariableReference> outputs) {
        return outputs.stream()
            .filter(v -> !inputs.containsKey(v))
            .collect(toList());
    }

    @Override
    public <T> T getInput(VariableReference input) {
        return (T) inputCache.get(input);
    }

    private void cacheInputs(Map<VariableReference, ?> inputs) {
        inputs.forEach((inputLabel, inputValue) -> inputCache.put(inputLabel, inputValue));
    }

    private Map<VariableReference, Object> convertToKeanuTensors(Collection<VariableReference> outputs,
                                                                 List<Tensor<?>> tfResults) {

        Map<VariableReference, Object> results = new HashMap<>();
        Iterator<Tensor<?>> resultIterator = tfResults.iterator();
        for (VariableReference output : outputs) {
            results.put(output, convertToKeanuTensor(resultIterator.next()));
        }
        return results;
    }

    private <T> T convertToKeanuTensor(Tensor<?> tensor) {

        try (Tensor<?> tfResult = tensor) {
            switch (tfResult.dataType()) {
                case DOUBLE:
                    return (T) toDoubleTensor(tfResult);
                case BOOL:
                    return (T) toBooleanTensor(tfResult);
                case INT32:
                    return (T) toIntegerTensor(tfResult);
                default:
                    throw new IllegalArgumentException("Cannot fetch output of type " + tfResult.dataType());
            }
        }
    }

    private Session.Runner feedInputs(Map<VariableReference, ?> inputs) {

        Session.Runner runner = session.runner();
        for (Map.Entry<VariableReference, ?> inputEntry : inputs.entrySet()) {

            Object tensor = inputEntry.getValue();

            Tensor<?> tensorFlowTensor = null;
            if (tensor instanceof DoubleTensor) {
                tensorFlowTensor = toTensorFlow((DoubleTensor) tensor);
            } else if (tensor instanceof BooleanTensor) {
                tensorFlowTensor = toTensorFlow((BooleanTensor) tensor);
            } else if (tensor instanceof IntegerTensor) {
                tensorFlowTensor = toTensorFlow((IntegerTensor) tensor);
            }

            runner = runner.feed(inputEntry.getKey().toStringReference(), tensorFlowTensor);
        }

        return runner;
    }

    private Session.Runner fetchOutputs(Session.Runner runner, Collection<VariableReference> outputs) {

        Session.Runner fetchedRunner = runner;
        for (VariableReference output : outputs) {
            fetchedRunner = fetchedRunner.fetch(output.toStringReference());
        }

        return fetchedRunner;
    }

    @Override
    public void close() {
        session.close();
    }
}
