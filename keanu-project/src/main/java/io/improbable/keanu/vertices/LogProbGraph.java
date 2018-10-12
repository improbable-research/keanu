package io.improbable.keanu.vertices;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class LogProbGraph {

    private final Map<Vertex<?>, Vertex<?>> inputs;
    private final DoubleVertex logProbOutput;

    public LogProbGraph(DoubleVertex logProbOutput) {
        this.inputs = new HashMap<>();
        this.logProbOutput = logProbOutput;
    }

    public <T> LogProbGraph addInput(Vertex<T> from, Vertex<T> to) {
        inputs.put(from, to);
        return this;
    }

    public Map<Vertex<?>, Vertex<?>> getInputs() {
        return inputs;
    }

    public <T> Vertex<T> getInput(Vertex<T> input) {
        return (Vertex<T>) inputs.get(input);
    }

    public DoubleVertex getLogProbOutput() {
        return logProbOutput;
    }
}
