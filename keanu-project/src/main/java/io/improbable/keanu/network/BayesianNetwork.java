package io.improbable.keanu.network;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import io.improbable.keanu.KeanuRandom;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.NonSaveableVertex;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.VertexLabel;

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

    private final List<? extends IVertex> vertices;
    private final Map<VertexLabel, IVertex> vertexLabels;
    private static final int TOP_LEVEL_INDENTATION = 1;
    private int indentation = TOP_LEVEL_INDENTATION;

    public BayesianNetwork(Set<? extends IVertex> vertices) {
        Preconditions.checkArgument(!vertices.isEmpty(), "A bayesian network must contain at least one vertex");
        this.vertices = ImmutableList.copyOf(vertices);
        this.vertexLabels = buildLabelMap(vertices);
    }

    public BayesianNetwork(Collection<? extends IVertex> vertices) {
        this(new HashSet<>(vertices));
    }

    public IVertex getVertexByLabel(VertexLabel label) {
        Preconditions.checkArgument(vertexLabels.containsKey(label), String.format("Vertex with label %s was not found in BayesianNetwork.", label));
        return vertexLabels.get(label);
    }

    public List<IVertex> getVerticesInNamespace(String... namespace) {
        return vertices.stream()
            .filter(v -> v.getLabel() != null && v.getLabel().isInNamespace(namespace))
            .collect(Collectors.toList());
    }

    public List<IVertex> getVerticesIgnoringNamespace(String innerNamespace) {
        return vertices.stream()
            .filter(v -> v.getLabel() != null && v.getLabel().getUnqualifiedName().equals(innerNamespace))
            .collect(Collectors.toList());
    }

    private Map<VertexLabel, IVertex> buildLabelMap(Set<? extends IVertex> vertices) {
        Map<VertexLabel, IVertex> labelMap = new HashMap<>();
        for (IVertex v : vertices) {
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

    public int getVertexCount() {
        return getVertices().size();
    }

    public double getAverageVertexDegree() {
        return getVertices().stream().mapToDouble(IVertex::getDegree).average().getAsDouble();
    }

    public void setState(NetworkState state) {
        for (VariableReference reference : state.getVariableReferences()) {
            this.vertices.stream()
                .filter(v -> v.getId() == reference)
                .forEach(v -> v.setValue(state.get(reference)));
        }
    }

    /**
     * @return A list of all vertices in the network.
     */
    public List<IVertex> getAllVertices() {
        return Collections.unmodifiableList(vertices);
    }

    public List<? extends IVertex> getVertices() {
        return vertices;
    }

    private interface VertexFilter {
        boolean filter(boolean isProbabilistic, boolean isObserved, int indentation);
    }

    private List<IVertex> getFilteredVertexList(VertexFilter filter) {
        return vertices.stream()
            .filter(v -> filter.filter(v.isProbabilistic(), v.isObserved(), v.getIndentation()))
            .collect(Collectors.toList());
    }

    /**
     * @return All vertices that are latent or observed
     */
    public List<IVertex> getLatentOrObservedVertices() {
        return getLatentOrObservedVertices(Integer.MAX_VALUE);
    }

    public List<IVertex> getTopLevelLatentOrObservedVertices() {
        return getLatentOrObservedVertices(TOP_LEVEL_INDENTATION);
    }

    private List<IVertex> getLatentOrObservedVertices(int maxIndentation) {
        return getFilteredVertexList((isProbabilistic, isObserved, indentation)
            -> (isProbabilistic || isObserved) && maxIndentation >= indentation);
    }

    /**
     * @return All vertices that are latent (i.e. probabilistic non-observed)
     */
    public List<IVertex> getLatentVertices() {
        return getLatentVertices(Integer.MAX_VALUE);
    }

    public List<IVertex> getTopLevelLatentVertices() {
        return getLatentVertices(TOP_LEVEL_INDENTATION);
    }

    private List<IVertex> getLatentVertices(int maxIndentation) {
        return getFilteredVertexList((isProbabilistic, isObserved, indentation)
            -> (isProbabilistic && !isObserved) && maxIndentation >= indentation);
    }

    /**
     * @return All vertices that are observed - which may be probabilistic or non-probabilistic
     */
    public List<IVertex> getObservedVertices() {
        return getObservedVertices(Integer.MAX_VALUE);
    }

    public List<IVertex> getTopLevelObservedVertices() {
        return getObservedVertices(TOP_LEVEL_INDENTATION);
    }

    private List<IVertex> getObservedVertices(int maxIndentation) {
        return getFilteredVertexList((isProbabilistic, isObserved, indentation) ->
            isObserved && maxIndentation >= indentation);
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

            List<IVertex> sortedByDependency = TopologicalSort.sort(getLatentVertices());
            setFromSampleAndCascade(sortedByDependency, random);

            probeForNonZeroProbability(sortedByDependency, attempts, random);
        }
    }

    /**
     * Attempt to find a non-zero master probability by repeatedly
     * cascading values from the given vertices
     */
    private void probeForNonZeroProbability(List<? extends IVertex> latentVertices, int attempts, KeanuRandom random) {

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
        return ProbabilityCalculator.isImpossibleLogProb(getLogOfMasterP());
    }

    public static void setFromSampleAndCascade(List<? extends IVertex> vertices) {
        setFromSampleAndCascade(vertices, KeanuRandom.getDefaultRandom());
    }

    public static void setFromSampleAndCascade(List<? extends IVertex> vertices, KeanuRandom random) {
        for (IVertex<?> vertex : vertices) {
            if (!(vertex instanceof Probabilistic)) {
                throw new IllegalArgumentException("Cannot sample from a non-probabilistic vertex. Vertex is: " + vertex);
            }
            setValueFromSample(vertex, random);
        }
        VertexValuePropagation.cascadeUpdate(vertices);
    }

    private static <T> void setValueFromSample(IVertex<T> vertex, KeanuRandom random) {
        vertex.setValue(((Probabilistic<T>) vertex).sample(random));
    }

    public List<IVertex<DoubleTensor>> getContinuousLatentVertices() {
        return getLatentVertices().stream()
            .filter(v -> v.getValue() instanceof DoubleTensor)
            .map(v -> (IVertex<DoubleTensor>) v)
            .collect(Collectors.toList());
    }

    public List<IVertex> getDiscreteLatentVertices() {
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
        if (isSaveable()) {
            for (IVertex vertex : TopologicalSort.sort(vertices)) {
                vertex.save(networkSaver);
            }
        } else {
            throw new IllegalArgumentException("Trying to save a BayesianNetwork that isn't Saveable");
        }
    }

    private boolean isSaveable() {
        return vertices.stream().filter(v -> v instanceof NonSaveableVertex).count() == 0;
    }

    public void saveValues(NetworkSaver networkSaver) {
        for (IVertex vertex : TopologicalSort.sort(vertices)) {
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
    public Set<IVertex> getSubgraph(IVertex vertex, int degree) {

        Set<IVertex> subgraphVertices = new HashSet<>();
        List<IVertex> verticesToProcessNow = new ArrayList<>();
        verticesToProcessNow.add(vertex);
        subgraphVertices.add(vertex);

        for (int distance = 0; distance < degree && !verticesToProcessNow.isEmpty(); distance++) {
            List<IVertex> connectedVertices = new ArrayList<>();

            for (IVertex v : verticesToProcessNow) {
                Stream<IVertex> verticesToAdd = Stream.concat(v.getParents().stream(), v.getChildren().stream());
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
