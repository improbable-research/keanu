package io.improbable.keanu.network;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import com.google.common.collect.ImmutableList;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class BayesianNetwork {

    private final List<? extends Vertex> vertices;
    private final Map<VertexLabel, Vertex> vertexLabels;
    private int depth = 1;

    public BayesianNetwork(Set<? extends Vertex> vertices) {
        this.vertices = ImmutableList.copyOf(vertices);
        this.vertexLabels = buildLabelMap(vertices);
    }

    public BayesianNetwork(Collection<? extends Vertex> vertices) {
        this(new HashSet<>(vertices));
    }

    public Vertex getVertexByLabel(VertexLabel label) {
        return vertexLabels.get(label);
    }

    private static Map<VertexLabel, Vertex> buildLabelMap(Set<? extends Vertex> vertices) {
        Map<VertexLabel, Vertex> labelMap = new HashMap<>();
        for (Vertex v : vertices) {
            VertexLabel label = v.getLabel();
            if (label != null) {
                if (labelMap.containsKey(label)) {
                    throw new IllegalArgumentException("Vertex Label Repeated: " + label);
                } else {
                    labelMap.put(label, v);
                }
            }
        }

        return labelMap;
    }

    List<? extends Vertex> getAllVertices() {
        return vertices;
    }

    public List<Vertex> getLatentAndObservedVertices() {
        return vertices.stream()
            .filter(v -> v.isProbabilistic() || v.isObserved())
            .collect(Collectors.toList());
    }

    /**
     * @return All vertices that are latent (i.e. probabilistic non-observed)
     */
    public List<Vertex> getLatentVertices() {
        return vertices.stream()
            .filter(v -> v.isProbabilistic() && !v.isObserved())
            .collect(Collectors.toList());
    }

    /**
     * @return All vertices that are observed - which may be probabilistic or non-probabilistic
     */
    public List<Vertex> getObservedVertices() {
        return vertices.stream()
            .filter(Vertex::isObserved)
            .collect(Collectors.toList());
    }

    public double getLogOfMasterP() {
        return ProbabilityCalculator.calculateLogProbFor(getLatentAndObservedVertices());
    }

    public void cascadeObservations() {
        VertexValuePropagation.cascadeUpdate(getObservedVertices());
    }


    public void probeForNonZeroProbability(int attempts) {
        probeForNonZeroProbability(attempts, KeanuRandom.getDefaultRandom());
    }

    /**
     * Attempt to find a non-zero master probability
     * by naively sampling vertices in order of data dependency
     *
     * @param attempts sampling attempts to get non-zero probability
     * @param random   random source for sampling
     */
    public void probeForNonZeroProbability(int attempts, KeanuRandom random) {

        if (isInImpossibleState()) {

            List<Vertex> sortedByDependency = TopologicalSort.sort(getLatentVertices());
            setFromSampleAndCascade(sortedByDependency, random);

            probeForNonZeroProbability(sortedByDependency, attempts, random);
        }
    }

    /**
     * Attempt to find a non-zero master probability by repeatedly
     * cascading values from the given vertices
     */
    private void probeForNonZeroProbability(List<? extends Vertex> latentVertices, int attempts, KeanuRandom random) {

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
        return getLatentVertices().stream()
            .filter(v -> v.getValue() instanceof DoubleTensor)
            .map(v -> (Vertex<DoubleTensor>) v)
            .collect(Collectors.toList());
    }

    public List<Vertex> getDiscreteLatentVertices() {
        return getLatentVertices().stream()
            .filter(v -> !(v.getValue() instanceof DoubleTensor))
            .collect(Collectors.toList());
    }

    public int getDepth() {
        return depth;
    }

    public void incrementDepth() {
        depth++;
    }

}
