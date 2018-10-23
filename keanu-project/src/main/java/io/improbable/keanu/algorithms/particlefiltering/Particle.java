package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Particle {

    private Map<Vertex, Object> latentVertices = new HashMap<>();
    private List<Vertex> observedVertices = new ArrayList<>();
    private double sumLogPOfSubgraph = 1.0;

    public Map<Vertex, Object> getLatentVertices() {
        return latentVertices;
    }

    public double getSumLogPOfSubgraph() {
        return sumLogPOfSubgraph;
    }

    public <T> void addLatentVertex(Vertex<T> vertex, T value) {
        latentVertices.put(vertex, value);
    }

    public <T> void addObservedVertex(Vertex<T> vertex) {
        observedVertices.add(vertex);
    }

    public double updateSumLogPOfSubgraph() {
        applyLatentVertexValues();
        double sumLogPOfLatents = ProbabilityCalculator.calculateLogProbFor(latentVertices.keySet());
        double sumLogPOfObservables = ProbabilityCalculator.calculateLogProbFor(observedVertices);
        sumLogPOfSubgraph = sumLogPOfLatents + sumLogPOfObservables;
        return sumLogPOfSubgraph;
    }

    public Particle shallowCopy() {
        Particle clone = new Particle();
        clone.latentVertices = new HashMap<>(this.latentVertices);
        clone.observedVertices = new ArrayList<>(this.observedVertices);
        return clone;
    }

    public static int sortDescending(Particle a, Particle b) {
        return Double.compare(b.getSumLogPOfSubgraph(), a.getSumLogPOfSubgraph());
    }

    private void applyLatentVertexValues() {
        latentVertices.keySet().forEach(this::applyLatentVertexValue);
    }

    private <T> void applyLatentVertexValue(Vertex<T> vertex) {
        if (latentVertices.containsKey(vertex)) {
            T value = (T) latentVertices.get(vertex);
            vertex.setAndCascade(value);
        }
    }
}
