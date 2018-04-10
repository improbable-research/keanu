package io.improbable.keanu.network;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.vertices.Vertex;

import java.util.*;
import java.util.stream.Collectors;

/**
 * A wrapper around a collection of connected vertices.
 */
public class BayesNet {

    private final List<Vertex<?>> verticesThatContributeToMasterP;
    private final List<Vertex<?>> latentVertices;
    private final List<Vertex<?>> observedVertices;

    //Lazy evaluated
    private List<Vertex<Double>> continuousLatentVertices;
    private List<Vertex<?>> discreteLatentVertices;

    public BayesNet(Set<? extends Vertex<?>> vertices) {

        verticesThatContributeToMasterP = vertices.stream()
                .filter(v -> v.isObserved() || v.isProbabilistic())
                .collect(Collectors.toList());

        observedVertices = verticesThatContributeToMasterP.stream()
                .filter(Vertex::isObserved)
                .collect(Collectors.toList());

        latentVertices = verticesThatContributeToMasterP.stream()
                .filter(v -> !v.isObserved())
                .collect(Collectors.toList());
    }

    public BayesNet(Collection<? extends Vertex<?>> vertices) {
        this(new HashSet<>(vertices));
    }

    public List<Vertex<?>> getVerticesThatContributeToMasterP() {
        return verticesThatContributeToMasterP;
    }

    public List<Vertex<?>> getLatentVertices() {
        return latentVertices;
    }

    public List<Vertex<Double>> getContinuousLatentVertices() {
        if (continuousLatentVertices == null) {
            splitContinuousAndDiscrete();
        }

        return continuousLatentVertices;
    }

    public List<Vertex<?>> getDiscreteLatentVertices() {
        if (discreteLatentVertices == null) {
            splitContinuousAndDiscrete();
        }

        return discreteLatentVertices;
    }

    public List<Vertex<?>> getObservedVertices() {
        return observedVertices;
    }

    private void splitContinuousAndDiscrete() {

        continuousLatentVertices = new ArrayList<>();
        discreteLatentVertices = new ArrayList<>();

        for (Vertex<?> vertex : latentVertices) {
            if (vertex.getValue() instanceof Double) {
                continuousLatentVertices.add((Vertex<Double>) vertex);
            } else {
                discreteLatentVertices.add(vertex);
            }
        }
    }

    public double getLogOfMasterP() {
        double sum = 0.0;
        for (Vertex<?> vertex : verticesThatContributeToMasterP) {
            sum += vertex.logDensityAtValue();
        }
        return sum;
    }

    /**
     * Attempt to find a non-zero master probability
     * by naively sampling vertices in order of data dependency
     *
     */
    public void probeForNonZeroMasterP(int attempts) {

        cascadeValues(observedVertices);
        List<? extends Vertex<?>> sortedByDependency = TopologicalSort.sort(latentVertices);
        sampleAndCascade(sortedByDependency);

        probeForNonZeroMasterP(sortedByDependency, attempts);
    }

    /**
     * Attempt to find a non-zero master probability by repeatedly
     * cascading values from the given vertices
     */
    private void probeForNonZeroMasterP(List<? extends Vertex<?>> latentVertices, int attempts) {

        int iteration = 0;
        while (isInImpossibleState()) {
            sampleAndCascade(latentVertices);
            iteration++;

            if (iteration > attempts) {
                throw new RuntimeException("Failed to find non-zero probability state");
            }
        }
    }

    public boolean isInImpossibleState() {
        double logOfMasterP = getLogOfMasterP();
        return logOfMasterP == Double.NEGATIVE_INFINITY || logOfMasterP == Double.NaN;
    }

    public static void sampleAndCascade(List<? extends Vertex<?>> vertices) {
        vertices.forEach(BayesNet::sampleAndCascade);
    }

    public static <T> void sampleAndCascade(Vertex<T> v) {
        v.setAndCascade(v.sample());
    }

    public static void cascadeValues(List<? extends Vertex<?>> vertices) {
        vertices.forEach(BayesNet::cascadeValue);
    }

    public static <T> void cascadeValue(Vertex<T> v) {
        v.updateChildren();
    }
}
