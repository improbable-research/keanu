package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.*;
import java.util.stream.Collectors;

public class BayesianNetwork {

    private final List<Vertex> latentAndObservedVertices;
    private final List<Vertex> latentVertices;
    private final List<Vertex> observedVertices;

    //Lazy evaluated
    private List<Vertex<DoubleTensor>> continuousLatentVertices;
    private List<Vertex> discreteLatentVertices;

    public BayesianNetwork(Set<? extends Vertex> vertices) {

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

    public BayesianNetwork(Collection<? extends Vertex> vertices) {
        this(new HashSet<>(vertices));
    }

    public List<Vertex> getLatentAndObservedVertices() {
        return latentAndObservedVertices;
    }

    public List<Vertex> getLatentVertices() {
        return latentVertices;
    }

    public List<Vertex> getObservedVertices() {
        return observedVertices;
    }

    public double getLogOfMasterP() {
        double sum = 0.0;
        for (Vertex<?> vertex : latentAndObservedVertices) {
            sum += vertex.logProbAtValue();
        }
        return sum;
    }

    public void cascadeObservations() {
        VertexValuePropagation.cascadeUpdate(observedVertices);
    }


    public void probeForNonZeroMasterP(int attempts) {
        probeForNonZeroMasterP(attempts, KeanuRandom.getDefaultRandom());
    }

    /**
     * Attempt to find a non-zero master probability
     * by naively sampling vertices in order of data dependency
     *
     * @param attempts sampling attempts to get non-zero probability
     * @param random   random source for sampling
     */
    public void probeForNonZeroMasterP(int attempts, KeanuRandom random) {

        if (isInImpossibleState()) {

            List<Vertex> sortedByDependency = TopologicalSort.sort(latentVertices);
            setFromSampleAndCascade(sortedByDependency, random);

            probeForNonZeroMasterP(sortedByDependency, attempts, random);
        }
    }

    /**
     * Attempt to find a non-zero master probability by repeatedly
     * cascading values from the given vertices
     */
    private void probeForNonZeroMasterP(List<? extends Vertex> latentVertices, int attempts, KeanuRandom random) {

        int iteration = 0;
        while (isInImpossibleState()) {
            setFromSampleAndCascade(latentVertices, random);
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

    public static void setFromSampleAndCascade(List<? extends Vertex> vertices) {
        setFromSampleAndCascade(vertices, KeanuRandom.getDefaultRandom());
    }

    public static void setFromSampleAndCascade(List<? extends Vertex> vertices, KeanuRandom random) {
        for (Vertex<?> vertex : vertices) {
            setValueFromSample(vertex, random);
        }
        VertexValuePropagation.cascadeUpdate(vertices);
    }

    private static <T> void setValueFromSample(Vertex<T> vertex, KeanuRandom random) {
        vertex.setValue(vertex.sample(random));
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

    private void splitContinuousAndDiscrete() {

        continuousLatentVertices = new ArrayList<>();
        discreteLatentVertices = new ArrayList<>();

        for (Vertex<?> vertex : getLatentVertices()) {
            if (vertex.getValue() instanceof DoubleTensor) {
                continuousLatentVertices.add((Vertex<DoubleTensor>) vertex);
            } else {
                discreteLatentVertices.add(vertex);
            }
        }
    }

}
