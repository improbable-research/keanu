package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleModelResultVertex;
import org.omg.SendingContext.RunTime;

import java.io.*;
import java.util.*;
import java.util.function.Function;
import java.util.regex.Pattern;

public class ShellModelVertex extends DoubleVertex implements ModelProcessVertex<DoubleTensor> {

    private String command;
    private Map<VertexLabel, DoubleVertex> inputs;
    private Map<VertexLabel, Double> outputs;
    private Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> extractOutput;

    public ShellModelVertex(String command,
                            Map<VertexLabel, DoubleVertex> inputs,
                            Map<VertexLabel, Double> outputs,
                            Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> extractOutput) {
        this.command = command;
        this.inputs = inputs;
        this.outputs = outputs;
        this.extractOutput = extractOutput;
        setParents(convertMapToList(inputs));
    }

    @Override
    public String process(Map<VertexLabel, DoubleVertex> inputs) {
        String newCommand = command;
        for (Map.Entry<VertexLabel, DoubleVertex> input : inputs.entrySet()) {
            String argument = "{" + input.getKey().toString() + "}";
            newCommand = newCommand.replaceAll(Pattern.quote(argument), input.getValue().getValue().scalar().toString());
        }
        return newCommand;
    }

    @Override
    public Map<VertexLabel, Double> run(String process, Function<Map<VertexLabel, DoubleVertex>, Map<VertexLabel, Double>> extractOutput) {
        try {
            Process cmd = Runtime.getRuntime().exec(process);
            cmd.waitFor();
            outputs = extractOutput.apply(inputs);
            return outputs;
        } catch (IOException | InterruptedException e) {
            throw new IllegalArgumentException("Failed to run model while executing the process.");
        }
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
    public DoubleTensor calculate() {
        run(process(inputs), extractOutput);
        return DoubleTensor.scalar(0.);
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return calculate();
    }

    @Override
    public void observe(DoubleTensor value) {
        throw new UnsupportedOperationException("Observing a Model Vertex is not supported.");
    }

    @Override
    public void unobserve() {
        throw new UnsupportedOperationException("Un-observing a Model Vertex is not supported.");
    }

    @Override
    public Optional<DoubleTensor> getObservedValue() {
        return Optional.empty();
    }

    @Override
    public boolean isObserved() {
        return false;
    }

    private List convertMapToList(Map<VertexLabel, DoubleVertex> inputs) {
        List<Vertex> asList = new ArrayList<>();
        for (Map.Entry<VertexLabel, DoubleVertex> input : inputs.entrySet()) {
            asList.add(input.getValue());
        }
        return asList;
    }
}
