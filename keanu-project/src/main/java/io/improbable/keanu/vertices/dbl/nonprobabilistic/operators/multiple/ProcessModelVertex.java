package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
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

/**
 * A vertex whose operation is the execution of an external process.
 *
 * It is able to execute an external process and is able to parse the result.
 *
 * It stores multiple output values and a model result vertex is required to extract a specific value.
 */
public class ProcessModelVertex extends DoubleVertex implements ModelVertex<DoubleTensor> {

    private String command;
    private Map<VertexLabel, Vertex<? extends Tensor>> inputs;
    private Map<VertexLabel, Tensor> outputs;
    private BiFunction<Map<VertexLabel, Vertex<? extends Tensor>>, String, String> commandFormatter;
    private Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Tensor>> extractOutput;
    private boolean hasCalculated;

    public ProcessModelVertex(String command,
                              Map<VertexLabel, Vertex<? extends Tensor>> inputs,
                              BiFunction<Map<VertexLabel, Vertex<? extends Tensor>>, String, String> commandFormatter,
                              Function<Map<VertexLabel, Vertex<? extends Tensor>>, Map<VertexLabel, Tensor>> extractOutput) {
        this.command = command;
        this.inputs = inputs;
        this.outputs = Collections.EMPTY_MAP;
        this.commandFormatter = commandFormatter;
        this.extractOutput = extractOutput;
        this.hasCalculated = false;
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
            hasCalculated = true;
        } catch (IOException | InterruptedException e) {
            throw new IllegalArgumentException("Failed to run model while executing the process.");
        }
    }

    @Override
    public Map<VertexLabel, Tensor> updateValues(Map<VertexLabel, Vertex<? extends Tensor>> inputs) {
        outputs = extractOutput.apply(inputs);
        return outputs;
    }

    @Override
    public boolean hasCalculated() {
        return hasCalculated;
    }

    @Override
    public DoubleTensor getDoubleModelOutputValue(VertexLabel label) {
        return (DoubleTensor) outputs.get(label);
    }

    @Override
    public IntegerTensor getIntegerModelOutputValue(VertexLabel label) {
        return (IntegerTensor) outputs.get(label);
    }

    @Override
    public BooleanTensor getBooleanModelOutputValue(VertexLabel label) {
        return (BooleanTensor) outputs.get(label);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        for (Map.Entry<VertexLabel, Vertex<? extends Tensor>> input : inputs.entrySet()) {
            input.getValue().sample();
        }
        return calculate();
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
