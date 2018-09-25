package io.improbable.keanu.network;

import static java.util.stream.Collectors.toMap;

import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.Vertex;

public class KeanuComputationalGraph {

    private final Map<String, Vertex> inputs;
    private final Map<String, Vertex> outputs;

    public KeanuComputationalGraph() {
        this.inputs = new HashMap<>();
        this.outputs = new HashMap<>();
    }

    public void addInput(String inputName, Vertex input) {
        this.inputs.put(inputName, input);
    }

    public void addOutput(String outputName, Vertex output) {
        this.outputs.put(outputName, output);
    }

    public KeanuComputationalGraph setInput(String name, Object inputValue) {
        this.inputs.get(name).setValue(inputValue);
        return this;
    }

    public <T> T calculate(String outputName) {
        return (T) calculate().get(outputName);
    }

    public Map<String, Object> calculate() {

        VertexValuePropagation.cascadeUpdate(this.inputs.values());

        return outputs.keySet().stream()
            .collect(toMap(
                name -> name,
                name -> outputs.get(name).getValue()
            ));
    }
}
