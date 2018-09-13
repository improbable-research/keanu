package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleModelResultVertex;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public class ProcessModelVertex extends DoubleVertex implements ModelVertex<DoubleTensor> {

    private String command;
    private Map<VertexLabel, DoubleVertex> inputs;
    private Map<VertexLabel, Double> outputs;
    private BiFunction<Map<VertexLabel, DoubleVertex>, String, String> commandFormatter;
    private Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> extractOutput;

    public ProcessModelVertex(String command,
                              Map<VertexLabel, DoubleVertex> inputs,
                              BiFunction<Map<VertexLabel, DoubleVertex>, String, String> commandFormatter,
                              Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> extractOutput) {
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
    public void run(Map<VertexLabel, DoubleVertex> inputs) {
        String newCommand = commandFormatter.apply(inputs, command);
        try {
            Process cmd = Runtime.getRuntime().exec(newCommand);
            cmd.waitFor();
        } catch (IOException | InterruptedException e) {
            throw new IllegalArgumentException("Failed to run model while executing the process.");
        }
    }

    @Override
    public Map<VertexLabel, Double> updateValues(Map<VertexLabel, DoubleVertex> inputs) {
        outputs = extractOutput.apply(inputs);
        return outputs;
    }

    @Override
    public Double getModelOutputValue(VertexLabel label) {
        return outputs.get(label);
    }

    @Override
    public DoubleVertex getModelOutputVertex(VertexLabel label) {
        return new DoubleModelResultVertex(this, label);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        for (Map.Entry<VertexLabel, DoubleVertex> input : inputs.entrySet()) {
            input.getValue().sample();
        }
        return calculate();
    }

    public Map<VertexLabel, Double> setValue(Map<VertexLabel, Double> values) {
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
