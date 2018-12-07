package io.improbable.keanu.network;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.algorithms.variational.optimizer.ProbabilisticGraph;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class BayesianNetwork {

    private final List<? extends Vertex> vertices;
    private final Map<VertexLabel, Vertex> vertexLabels;
    private static final int TOP_LEVEL_INDENTATION = 1;
    private int indentation = TOP_LEVEL_INDENTATION;

    public BayesianNetwork(Set<? extends Vertex> vertices) {
        Preconditions.checkArgument(!vertices.isEmpty(), "A bayesian network must contain at least one vertex");
        this.vertices = ImmutableList.copyOf(vertices);
        this.vertexLabels = buildLabelMap(vertices);
    }

    public BayesianNetwork(Collection<? extends Vertex> vertices) {
        this(new HashSet<>(vertices));
    }

    public Vertex getVertexByLabel(VertexLabel label) {
        return vertexLabels.get(label);
    }

    private Map<VertexLabel, Vertex> buildLabelMap(Set<? extends Vertex> vertices) {
        Map<VertexLabel, Vertex> labelMap = new HashMap<>();
        for (Vertex v : vertices) {
            VertexLabel label = v.getLabel();
            if (v.getIndentation() == this.indentation && label != null) {
                if (labelMap.containsKey(label)) {
                    throw new IllegalArgumentException("Vertex Label Repeated: " + label);
                } else {
                    labelMap.put(label, v);
                }
            }
        }

        return labelMap;
    }

    List<? extends Vertex> getVertices() {
        return vertices;
    }

    public int getVertexCount() {
        return getVertices().size();
    }

    public double getAverageVertexDegree() {
        return getVertices().stream().mapToDouble(Vertex::getDegree).average().getAsDouble();
    }

    public void setState(NetworkState state) {
        for (VertexId vertexId : state.getVertexIds()) {
            this.vertices.stream()
                .filter(v -> v.getId() == vertexId)
                .forEach(v -> v.setValue(state.get(vertexId)));
        }
    }

    /**
     * @return A list of all vertices in the network.
     */
    public List<Vertex> getAllVertices() {
        return Collections.unmodifiableList(vertices);
    }

    private interface VertexFilter {
        boolean filter(boolean isProbabilistic, boolean isObserved, int indentation);
    }

    private List<Vertex> getFilteredVertexList(VertexFilter filter) {
        return vertices.stream()
            .filter(v -> filter.filter(v.isProbabilistic(), v.isObserved(), v.getIndentation()))
            .collect(Collectors.toList());
    }

    /**
     * @return All vertices that are latent or observed
     */
    public List<Vertex> getLatentOrObservedVertices() {
        return getLatentOrObservedVertices(Integer.MAX_VALUE);
    }

    public List<Vertex> getTopLevelLatentOrObservedVertices() {
        return getLatentOrObservedVertices(TOP_LEVEL_INDENTATION);
    }

    private List<Vertex> getLatentOrObservedVertices(int maxIndentation) {
        return getFilteredVertexList((isProbabilistic, isObserved, indentation)
            -> (isProbabilistic || isObserved) && maxIndentation >= indentation);
    }

    /**
     * @return All vertices that are latent (i.e. probabilistic non-observed)
     */
    public List<Vertex> getLatentVertices() {
        return getLatentVertices(Integer.MAX_VALUE);
    }

    public List<Vertex> getTopLevelLatentVertices() {
        return getLatentVertices(TOP_LEVEL_INDENTATION);
    }

    private List<Vertex> getLatentVertices(int maxIndentation) {
        return getFilteredVertexList((isProbabilistic, isObserved, indentation)
            -> (isProbabilistic && !isObserved) && maxIndentation >= indentation);
    }

    /**
     * @return All vertices that are observed - which may be probabilistic or non-probabilistic
     */
    public List<Vertex> getObservedVertices() {
        return getObservedVertices(Integer.MAX_VALUE);
    }

    public List<Vertex> getTopLevelObservedVertices() {
        return getObservedVertices(TOP_LEVEL_INDENTATION);
    }

    private List<Vertex> getObservedVertices(int maxIndentation) {
        return getFilteredVertexList((isProbabilistic, isObserved, indentation) ->
            isObserved && maxIndentation >= indentation);
    }

    /**
     * @return a list of all vertices that are not differentiable (i.e., there are points at which they do not have a derivative).
     */
    public List<Vertex> getNonDifferentiableVertices() {
        return vertices.stream().filter(vertex -> !(vertex instanceof Differentiable))
            .collect(Collectors.toList());
    }

    public double getLogOfMasterP() {
        return ProbabilityCalculator.calculateLogProbFor(getLatentOrObservedVertices());
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
        return ProbabilisticGraph.isImpossible(getLogOfMasterP());
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

    public int getIndentation() {
        return indentation;
    }

    public void incrementIndentation() {
        indentation++;
    }

    public void save(NetworkSaver networkSaver) {
        if(isSaveable()) {
            for (Vertex vertex : TopologicalSort.sort(vertices)) {
                vertex.save(networkSaver);
            }
        }
        else {
            throw new IllegalArgumentException("Trying to save a BayesianNetwork that isn't Saveable");
        }
    }

    private boolean isSaveable() {
        return vertices.stream().filter(v -> v instanceof NonSaveableVertex).count() == 0;
    }

    public void saveValues(NetworkSaver networkSaver) {
        for (Vertex vertex : vertices) {
            vertex.saveValue(networkSaver);
        }
    }

    /**
     * Method for traversing a graph and returning a subgraph of vertices within the given degree of the specified vertex.
     *
     * @param vertex vertex that the subgraph will be centered around
     * @param degree degree of connections from the vertex to be included in the subgraph
     * @return a set of vertices within the specified degree from the given vertex
     */
    public Set<Vertex> getSubgraph(Vertex vertex, int degree) {

        Set<Vertex> subgraphVertices = new HashSet<>();
        List<Vertex> verticesToProcessNow = new ArrayList<>();
        verticesToProcessNow.add(vertex);
        subgraphVertices.add(vertex);

        for (int distance = 0; distance < degree && !verticesToProcessNow.isEmpty(); distance++) {
            List<Vertex> connectedVertices = new ArrayList<>();

            for (Vertex v : verticesToProcessNow) {
                Stream<Vertex> verticesToAdd = Stream.concat(v.getParents().stream(), v.getChildren().stream());
                verticesToAdd
                    .filter(a -> !subgraphVertices.contains(a))
                    .forEachOrdered(a -> {
                        connectedVertices.add(a);
                        subgraphVertices.add(a);
                    });
            }

            verticesToProcessNow = connectedVertices;
        }

        return subgraphVertices;
    }
}
