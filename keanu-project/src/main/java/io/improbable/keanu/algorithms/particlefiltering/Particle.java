package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.ProbabilityCalculator;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents a Particle used in a Particle Filter.
 * A Particle can be thought of as a particular instance of the network with state and an associated probability.
 */
public class Particle {

    private Map<IVertex, Object> latentVertices = new HashMap<>();
    private List<IVertex> observedVertices = new ArrayList<>();
    private double sumLogPOfSubgraph = 1.0;

    public Map<IVertex, Object> getLatentVertices() {
        return latentVertices;
    }

    /**
     * @return log probability of the subgraph occuring in it's current state
     */
    public double logProb() {
        return sumLogPOfSubgraph;
    }

    public double getScalarValueOfVertex(IVertex<DoubleTensor> vertex) {
        return ((DoubleTensor) latentVertices.get(vertex)).scalar();
    }

    public <T> T getValueOfVertex(IVertex<T> vertex) {
        return (T) latentVertices.get(vertex);
    }

    <T> void addLatentVertex(IVertex<T> vertex, T value) {
        latentVertices.put(vertex, value);
    }

    <T> void addObservedVertex(IVertex<T> vertex) {
        observedVertices.add(vertex);
    }

    double updateSumLogPOfSubgraph() {
        applyLatentVertexValues();
        double sumLogPOfLatents = ProbabilityCalculator.calculateLogProbFor(latentVertices.keySet());
        double sumLogPOfObservables = ProbabilityCalculator.calculateLogProbFor(observedVertices);
        sumLogPOfSubgraph = sumLogPOfLatents + sumLogPOfObservables;
        return sumLogPOfSubgraph;
    }

    Particle shallowCopy() {
        Particle clone = new Particle();
        clone.latentVertices = new HashMap<>(this.latentVertices);
        clone.observedVertices = new ArrayList<>(this.observedVertices);
        return clone;
    }

    static int sortDescending(Particle a, Particle b) {
        return Double.compare(b.logProb(), a.logProb());
    }

    private void applyLatentVertexValues() {
        latentVertices.keySet().forEach(this::applyLatentVertexValue);
    }

    private <T> void applyLatentVertexValue(IVertex<T> vertex) {
        if (latentVertices.containsKey(vertex)) {
            T value = (T) latentVertices.get(vertex);
            vertex.setAndCascade(value);
        }
    }
}
