package io.improbable.keanu.network;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.KeanuSavedBayesNet;
import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.algorithms.graphtraversal.VertexValuePropagation;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ProbabilityCalculator;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class BayesianNetwork {

    private final List<? extends Vertex> vertices;
    private final Map<VertexLabel, Vertex> vertexLabels;
    private final int TOP_LEVEL_INDENTATION = 1;
    private int indentation = TOP_LEVEL_INDENTATION;

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
        double logOfMasterP = getLogOfMasterP();
        return logOfMasterP == Double.NEGATIVE_INFINITY || Double.isNaN(logOfMasterP);
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

    public void save(ProtobufWriter protobufWriter) throws IOException {
        for (Vertex vertex : TopologicalSort.sort(vertices)) {
            vertex.saveTo(protobufWriter);
        }
    }

    public void saveValues(ProtobufWriter protobufWriter) throws IOException {
        for (Vertex vertex : vertices) {
            vertex.saveValueTo(protobufWriter);
        }
    }

    public static BayesianNetwork loadNetwork(InputStream input) throws IOException {
        Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices = new HashMap<>();
        KeanuSavedBayesNet.BayesianNetwork parsedNet = KeanuSavedBayesNet.BayesianNetwork.parseFrom(input);
        List<KeanuSavedBayesNet.Vertex> sortedVertexList = new ArrayList<>(parsedNet.getVerticesList());
        sortedVertexList.sort(BayesianNetwork::compareIDs);

        for (KeanuSavedBayesNet.Vertex vertex : sortedVertexList) {
            Vertex newVertex = Vertex.fromProtoBuf(vertex, instantiatedVertices);
            instantiatedVertices.put(vertex.getId(), newVertex);
        }

        BayesianNetwork bayesNet = new BayesianNetwork(instantiatedVertices.values());

        loadDefaultValues(parsedNet, instantiatedVertices, bayesNet);

        return bayesNet;
    }

    private static int compareIDs(KeanuSavedBayesNet.Vertex v1, KeanuSavedBayesNet.Vertex v2) {
        VertexId vID1 = new VertexId(v1.getId().getIdValuesList());
        VertexId vID2 = new VertexId(v2.getId().getIdValuesList());

        return vID1.compareTo(vID2);
    }

    private static void loadDefaultValues(KeanuSavedBayesNet.BayesianNetwork parsedNet,
                                          Map<KeanuSavedBayesNet.VertexID, Vertex> instantiatedVertices,
                                          BayesianNetwork bayesNet) {
        for (KeanuSavedBayesNet.StoredValue value : parsedNet.getDefaultStateList()) {
            Vertex targetVertex = null;

            if (value.hasId()) {
                targetVertex = instantiatedVertices.get(value.getId());
            }

            if (value.getVertexLabel() != "") {
                Vertex newTarget = bayesNet.getVertexByLabel(new VertexLabel(value.getVertexLabel()));

                if (targetVertex != null && newTarget != targetVertex) {
                    throw new IllegalArgumentException("Label and VertexID don't refer to same Vertex");
                } else {
                    targetVertex = newTarget;
                }
            }

            if (targetVertex == null) {
                throw new IllegalArgumentException("Value specified for unknown Vertex");
            }

            targetVertex.setValue(value.getValue());
        }
    }
}
