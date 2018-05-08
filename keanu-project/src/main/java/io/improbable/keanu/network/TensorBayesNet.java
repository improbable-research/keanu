package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.*;
import java.util.stream.Collectors;

/**
 * A wrapper around a collection of connected vertices.
 */
public class TensorBayesNet {

    private final List<Vertex> latentAndObservedVertices;
    private final List<Vertex> latentVertices;
    private final List<Vertex> observedVertices;

    //Lazy evaluated
    private List<Vertex<DoubleTensor>> continuousLatentVertices;
    private List<Vertex> discreteLatentVertices;

    public TensorBayesNet(Set<? extends Vertex> vertices) {

        latentAndObservedVertices = vertices.stream()
                .filter(v -> v.isObserved() || v.isProbabilistic())
                .collect(Collectors.toList());

        observedVertices = latentAndObservedVertices.stream()
                .filter(Vertex::isObserved)
                .collect(Collectors.toList());

        latentVertices = latentAndObservedVertices.stream()
                .filter(v -> !v.isObserved())
                .collect(Collectors.toList());
    }

    public TensorBayesNet(Collection<? extends Vertex> vertices) {
        this(new HashSet<>(vertices));
    }

    public List<Vertex> getLatentAndObservedVertices() {
        return latentAndObservedVertices;
    }

    public List<Vertex> getLatentVertices() {
        return latentVertices;
    }

    public List<Vertex<DoubleTensor>> getContinuousLatentVertices() {
        if (continuousLatentVertices == null) {
            splitContinuousAndDiscrete();
        }

        return continuousLatentVertices;
    }

    public List<Vertex> getDiscreteLatentVertices() {
        if (discreteLatentVertices == null) {
            splitContinuousAndDiscrete();
        }

        return discreteLatentVertices;
    }

    public List<Vertex> getObservedVertices() {
        return observedVertices;
    }

    private void splitContinuousAndDiscrete() {

        continuousLatentVertices = new ArrayList<>();
        discreteLatentVertices = new ArrayList<>();

        for (Vertex<?> vertex : latentVertices) {
            if (vertex.getValue() instanceof DoubleTensor) {
                continuousLatentVertices.add((Vertex<DoubleTensor>)vertex);
            } else {
                discreteLatentVertices.add(vertex);
            }
        }
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
