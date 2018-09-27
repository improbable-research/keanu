package io.improbable.keanu.vertices;

import java.util.HashMap;
import java.util.Map;

public class LogProbGraph {

    private final Map<Vertex<?>, Vertex<?>> inputs;
    private final Vertex logProbOutput;

    public LogProbGraph(Vertex logProbOutput) {
        this.inputs = new HashMap<>();
        this.logProbOutput = logProbOutput;
    }

    public LogProbGraph addInput(Vertex from, Vertex to) {
        inputs.put(from, to);
        return this;
    }

    public Map<Vertex<?>, Vertex<?>> getInputs() {
        return inputs;
    }

    public Vertex getLogProbOutput() {
        return logProbOutput;
    }
}
