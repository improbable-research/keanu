package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.ContinuousVertex;
import io.improbable.keanu.vertices.Vertex;

import java.util.*;

/**
 * A wrapper around a collection of connected vertices.
 */
public class BayesNet {

    private final List<Vertex> latentAndObservedVertices;
    private final List<ContinuousVertex<Double>> latentAndObservedContinuousVertices;
    private final List<Vertex> latentAndObservedDiscreteVertices;

    private final List<Vertex> latentVertices;
    private List<ContinuousVertex<Double>> continuousLatentVertices;
    private List<Vertex> discreteLatentVertices;

    private final List<Vertex> observedVertices;
    private List<ContinuousVertex<Double>> continuousObservedVertices;
    private List<Vertex> discreteObservedVertices;

    public BayesNet(Set<? extends Vertex> vertices) {

        continuousLatentVertices = new ArrayList<>();
        discreteLatentVertices = new ArrayList<>();

        continuousObservedVertices = new ArrayList<>();
        discreteObservedVertices = new ArrayList<>();

        for (Vertex v : vertices) {
            if (v.isObserved()) {
                if (v instanceof ContinuousVertex) {
                    continuousObservedVertices.add((ContinuousVertex<Double>) v);
                } else {
                    discreteObservedVertices.add(v);
                }
            } else if (v.isProbabilistic()) {
                if (v instanceof ContinuousVertex) {
                    continuousLatentVertices.add((ContinuousVertex<Double>) v);
                } else {
                    discreteLatentVertices.add(v);
                }
            }
        }

        observedVertices = new ArrayList<>();
        observedVertices.addAll(continuousObservedVertices);
        observedVertices.addAll(discreteObservedVertices);

        latentVertices = new ArrayList<>();
        latentVertices.addAll(continuousLatentVertices);
        latentVertices.addAll(discreteLatentVertices);

        latentAndObservedVertices = new ArrayList<>();
        latentAndObservedVertices.addAll(latentVertices);
        latentAndObservedVertices.addAll(observedVertices);

        latentAndObservedContinuousVertices = new ArrayList<>();
        latentAndObservedContinuousVertices.addAll(continuousLatentVertices);
        latentAndObservedContinuousVertices.addAll(continuousObservedVertices);

        latentAndObservedDiscreteVertices = new ArrayList<>();
        latentAndObservedDiscreteVertices.addAll(discreteLatentVertices);
        latentAndObservedDiscreteVertices.addAll(discreteObservedVertices);
    }

    public BayesNet(Collection<? extends Vertex> vertices) {
        this(new HashSet<>(vertices));
    }

    public List<Vertex> getLatentAndObservedVertices() {
        return latentAndObservedVertices;
    }

    public List<ContinuousVertex<Double>> getLatentAndObservedContinuousVertices() {
        return latentAndObservedContinuousVertices;
    }

    public List<Vertex> getLatentAndObservedDiscreteVertices() {
        return latentAndObservedDiscreteVertices;
    }

    public List<Vertex> getLatentVertices() {
        return latentVertices;
    }

    public List<ContinuousVertex<Double>> getContinuousLatentVertices() {
        return continuousLatentVertices;
    }

    public List<Vertex> getDiscreteLatentVertices() {
        return discreteLatentVertices;
    }

    public List<Vertex> getObservedVertices() {
        return observedVertices;
    }

    public List<ContinuousVertex<Double>> getContinuousObservedVertices() {
        return continuousObservedVertices;
    }

    public List<Vertex> getDiscreteObservedVertices() {
        return discreteObservedVertices;
    }

    public double getLogOfMasterP() {
        double sum = 0.0;
        for (Vertex<?> vertex : latentAndObservedVertices) {
            sum += vertex.logProbAtValue();
        }
        return sum;
    }

    /**
     * Attempt to find a non-zero master probability
     * by naively sampling vertices in order of data dependency
     *
     * @param attempts sampling attempts to get non-zero probability
     */
    public void probeForNonZeroMasterP(int attempts) {

        VertexValuePropagation.cascadeUpdate(observedVertices);
        List<Vertex> sortedByDependency = TopologicalSort.sort(latentVertices);
        setFromSampleAndCascade(sortedByDependency);

        probeForNonZeroMasterP(sortedByDependency, attempts);
    }

    /**
     * Attempt to find a non-zero master probability by repeatedly
     * cascading values from the given vertices
     */
    private void probeForNonZeroMasterP(List<Vertex> latentVertices, int attempts) {

        Map<String, Long> setAndCascadeCache = VertexValuePropagation.exploreSetting(latentVertices);
        int iteration = 0;
        while (isInImpossibleState()) {
            setFromSampleAndCascade(latentVertices, setAndCascadeCache);
            iteration++;

            if (iteration > attempts) {
                throw new IllegalStateException("Failed to find non-zero probability state");
            }
        }
    }

    public boolean isInImpossibleState() {
        double logOfMasterP = getLogOfMasterP();
        return logOfMasterP == Double.NEGATIVE_INFINITY || logOfMasterP == Double.NaN;
    }

    public static void setFromSampleAndCascade(List<Vertex> vertices) {
        setFromSampleAndCascade(vertices, VertexValuePropagation.exploreSetting(vertices));
    }

    public static void setFromSampleAndCascade(List<Vertex> vertices, Map<String, Long> setAndCascadeCache) {
        for (Vertex<?> vertex : vertices) {
            setValueFromSample(vertex);
        }
        VertexValuePropagation.cascadeUpdate(vertices, setAndCascadeCache);
    }

    private static <T> void setValueFromSample(Vertex<T> vertex) {
        vertex.setValue(vertex.sample());
    }

}
