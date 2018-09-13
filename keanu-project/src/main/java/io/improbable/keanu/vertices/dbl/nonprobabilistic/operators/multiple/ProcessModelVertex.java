package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ProcessModelVertex extends DoubleVertex implements ModelVertex<DoubleTensor> {

    private String command;
    private Map<VertexLabel, Vertex<? extends Tensor>> inputs;
    private Map<VertexLabel, Object> outputs;
    private BiFunction<Map<VertexLabel, Vertex<? extends Tensor>>, String, String> commandFormatter;
    private Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Object>> extractOutput;

    public ProcessModelVertex(String command,
                              Map<VertexLabel, Vertex<? extends Tensor>> inputs,
                              BiFunction<Map<VertexLabel, Vertex<? extends Tensor>>, String, String> commandFormatter,
                              Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Object>> extractOutput) {
        this.command = command;
        this.inputs = inputs;
        this.outputs = Collections.EMPTY_MAP;
        this.commandFormatter = commandFormatter;
        this.extractOutput = extractOutput;
        setParents(inputs.entrySet().stream().map(r -> r.getValue()).collect(Collectors.toList()));
    }

    @Override
    public DoubleTensor calculate() {
        run(inputs);
        updateValues(inputs);
        return DoubleTensor.scalar(0.);
    }

    @Override
    public void run(Map<VertexLabel, Vertex<? extends Tensor>> inputs) {
        String newCommand = commandFormatter.apply(inputs, command);
        try {
            Process cmd = Runtime.getRuntime().exec(newCommand);
            cmd.waitFor();
        } catch (IOException | InterruptedException e) {
            throw new IllegalArgumentException("Failed to run model while executing the process.");
        }
    }

    @Override
    public Map<VertexLabel, Object> updateValues(Map<VertexLabel, Vertex<? extends Tensor>> inputs) {
        outputs = extractOutput.apply(inputs);
        return outputs;
    }

    @Override
    public Double getDoubleModelOutputValue(VertexLabel label) {
        return (Double) outputs.get(label);
    }

    @Override
    public Integer getIntegerModelOutputValue(VertexLabel label) {
        return (Integer) outputs.get(label);
    }

    @Override
    public Boolean getBooleanModelOutputValue(VertexLabel label) {
        return (Boolean) outputs.get(label);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        for (Map.Entry<VertexLabel, Vertex<? extends Tensor>> input : inputs.entrySet()) {
            input.getValue().sample();
        }
        return calculate();
    }

    public Map<VertexLabel, Object> setValue(Map<VertexLabel, Object> values) {
        outputs = values;
        return outputs;
    }

    @Override
    public Optional<DoubleTensor> getObservedValue() {
        return Optional.empty();
    }

    @Override
    public boolean isObserved() {
        return false;
    }

}
