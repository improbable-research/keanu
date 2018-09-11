package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import com.google.common.util.concurrent.UncheckedExecutionException;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleModelResultVertex;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.regex.Pattern;

public class ShellModelVertex extends DoubleVertex implements ModelVertex<DoubleTensor> {

    private String command;
    private Map<VertexLabel, DoubleVertex> inputs;
    private Map<VertexLabel, Double> outputs;
    private Function<Process, Map<VertexLabel, Double>> readInputs;

    public ShellModelVertex(String command,
                            Map<VertexLabel, DoubleVertex> inputs,
                            Map<VertexLabel, Double> outputs,
                            Function<Process, Map<VertexLabel, Double>> readInputs) {
        this.command = command;
        this.inputs = inputs;
        this.outputs = outputs;
        this.readInputs = readInputs;
        setParents(convertMapToList(inputs));
    }

    @Override
    public Map<VertexLabel, Double> run() {
        String newCommand = formatCommand();
        try {
            Process cmd = Runtime.getRuntime().exec(newCommand);
            cmd.waitFor();
            outputs = readInputs.apply(cmd);
            return outputs;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (InterruptedException e) {
            throw new UncheckedExecutionException(e);
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
        run();
        return null;
    }

    @Override
    public DoubleTensor sample(KeanuRandom random) {
        return calculate();
    }

    @Override
    public void observe(DoubleTensor value) {
        throw new UnsupportedOperationException("Cannot observe a Model");
    }

    @Override
    public void unobserve() {
        throw new UnsupportedOperationException("Cannot unobserve a Model");
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

    private String formatCommand() {
        String newCommand = command;
        for (Map.Entry<VertexLabel, DoubleVertex> input : inputs.entrySet()) {
            String argument = "{" + input.getKey().toString() + "}";
            newCommand = newCommand.replaceAll(Pattern.quote(argument), input.getValue().getValue().scalar().toString());
        }
        return newCommand;
    }
}
