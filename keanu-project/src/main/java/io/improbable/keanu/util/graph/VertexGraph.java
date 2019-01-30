package io.improbable.keanu.util.graph;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.awt.*;
import java.lang.reflect.Method;
import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;

/*
This class implements AbstractGraph that can be generated from a BayesianNetwork
 */
public class VertexGraph extends AbstractGraph<BasicGraphNode, BasicGraphEdge> {

    private static final Color PROBABILISTIC_COLOR = Color.PINK;
    private static final Color OBSERVED_COLOR = Color.BLUE;
    private static final Color DETERMINISTIC_COLOR = Color.CYAN;

    private static final String LABEL_FIELD = "label";
    private static final String COLOR_FIELD = "color";
    private static final Color DOUBLE_COLOR = Color.red;
    private static final Color INTEGER_COLOR = Color.green;
    private static final Color BOOLEAN_COLOR = Color.blue;
    private static final Color OTHER_COLOR = Color.darkGray;

    private Map<Vertex, BasicGraphNode> vertexNodes = new HashMap<>();
    private Map<BasicGraphNode, Vertex> invertedVertexNodes = new HashMap<>();

    public VertexGraph(BayesianNetwork inputNetwork) {
        this(inputNetwork.getAllVertices());
    }

    public VertexGraph(BayesianNetwork inputNetwork, Vertex v, int degree) {
        this(inputNetwork.getSubgraph(v, degree));
    }

    public VertexGraph(Vertex v) {
        this(v.getConnectedGraph());
    }

    public VertexGraph(Collection<Vertex> inputList) {
        // we loop twice first to create the nodes, then second time we know the other end should exist if it's in scope
        for (Vertex v : inputList) {
            addVertex(v);
        }
        for (Vertex v : inputList) {
            addVertexEdges(v);
        }
    }

    private static final String formatColorForDot(Color c) {
        return String.format("#%06X", (0xFFFFFF & c.getRGB()));
    }

    private void addVertex(Vertex v) {
        if (!vertexNodes.containsKey(v)) {
            BasicGraphNode n = createGraphNodeFor(v);
            vertexNodes.put(v, n);
            invertedVertexNodes.put(n, v);
        }
    }

    private BasicGraphNode createGraphNodeFor(Vertex v) {
        BasicGraphNode n = new BasicGraphNode();
        n.details.put(LABEL_FIELD, getBasicNameForVertex(v));
        return n;
    }

    private String getBasicNameForVertex(Vertex v) {
        if ( v.getLabel() != null ){
            return v.getLabel().getUnqualifiedName();
        }
        DisplayInformationForOutput vertexAnnotation = v.getClass().getAnnotation(DisplayInformationForOutput.class);
        if (vertexAnnotation != null && !vertexAnnotation.displayName().isEmpty()) {
            return vertexAnnotation.displayName();
        } else {
            return v.getClass().getSimpleName();
        }
    }

    private void addVertexEdges(Vertex v) {
        Set<Vertex> parents = v.getParents();
        BasicGraphNode dest = vertexNodes.get(v);
        for (Vertex other : parents) {
            if (vertexNodes.containsKey(other)) {
                BasicGraphNode src = vertexNodes.get(other);
                createGraphEdgeFor(src, dest);
            }
        }
    }

    private void createGraphEdgeFor(BasicGraphNode src, BasicGraphNode dest) {
        BasicGraphEdge e = new BasicGraphEdge(src, dest);
        edges.add(e);
    }

    public VertexGraph removeDeterministicVertices() {
        removeVerticesPreservingEdges((v) -> !v.isProbabilistic());
        return this;
    }

    public VertexGraph removeIntermediateVertices() {
        removeVerticesPreservingEdges((v) ->
            !v.isProbabilistic() && !(v instanceof ConstantDoubleVertex) && !v.getChildren().isEmpty());
        return this;
    }

    public void removeVerticesPreservingEdges(Predicate<Vertex> f) {
        Set<Vertex> toRemove = new HashSet<>();
        for (Vertex v : vertexNodes.keySet()) {
            if (f.test(v)) {
                toRemove.add(v);
            }
        }
        for (Vertex v : toRemove) {
            removeVertexPreservingEdges(v);
        }
    }

    public void removeVertex(Vertex v) {
        if (vertexNodes.containsKey(v)) {
            BasicGraphNode n = vertexNodes.remove(v);
            invertedVertexNodes.remove(n);
            Set<BasicGraphEdge> edgesToRemove = findEdgesUsing(n);
            edges.removeAll(edgesToRemove);
        }
    }

    public VertexGraph colorVerticesByState() {
        setVertexMetadata(COLOR_FIELD, this::convertStateToColor);
        return this;
    }


    public VertexGraph colorVerticesByType() {
        setVertexMetadata(COLOR_FIELD, this::convertTypeToColor);
        return this;
    }

    private String convertStateToColor(Vertex v) {
        Color c;
        if (v.isObserved()) {
            c = OBSERVED_COLOR;
        } else if (v instanceof Probabilistic) {
            c = PROBABILISTIC_COLOR;
        } else {
            c = DETERMINISTIC_COLOR;
        }
        return formatColorForDot(c);
    }

    private String convertTypeToColor(Vertex v) {
        Color c;
        if (v instanceof DoubleVertex) {
            c = DOUBLE_COLOR;
        } else if (v instanceof IntegerVertex) {
            c = INTEGER_COLOR;
        } else if (v instanceof BooleanVertex) {
            c = BOOLEAN_COLOR;
        } else {
            c = OTHER_COLOR;
        }
        return formatColorForDot(c);
    }

    public VertexGraph labelVerticesWithValue() {
        setVertexMetadata(LABEL_FIELD, this::convertValueToString);
        return this;
    }

    public VertexGraph labelEdgesWithParameters() {
        setEdgeMetadata(LABEL_FIELD, this::readEdgeName);
        return this;
    }

    private String convertValueToString(Vertex vertex) {
        Object obj = vertex.getValue();
        if (obj instanceof Tensor) {
            if (!((Tensor) obj).isScalar()) return null;
            return ((Tensor) obj).scalar().toString();
        } else {
            return obj.toString();
        }
    }

    private String readEdgeName(BasicGraphEdge edge) {
        Vertex vertex = invertedVertexNodes.get(edge.getDestination());
        Vertex srcVertex = invertedVertexNodes.get(edge.getSource());
        // Check if any of the edges represent a connection between the vertex and its hyperparameter and annotate it accordingly.
        Class vertexClass = vertex.getClass();
        Method[] methods = vertexClass.getMethods();

        for (Method method : methods) {
            SaveVertexParam annotation = method.getAnnotation(SaveVertexParam.class);
            if (annotation != null && Vertex.class.isAssignableFrom(method.getReturnType())) {
                String parentName = annotation.value();
                try {
                    Vertex parentVertex = (Vertex) method.invoke(vertex);
                    if (parentVertex == srcVertex) {
                        return parentName;
                    }
                } catch (Exception e) {
                    throw new IllegalArgumentException("Invalid parent retrieval function specified", e);
                }
            }
        }
        return "?";
    }

    public void setVertexMetadata(String field, Function<Vertex, String> f) {
        for (Map.Entry<Vertex, BasicGraphNode> e : vertexNodes.entrySet()) {
            String v = f.apply(e.getKey());
            if (v != null) e.getValue().details.put(field, v);
        }
    }

    public void setEdgeMetadata(String field, Function<BasicGraphEdge, String> f) {
        for (BasicGraphEdge e : edges) {
            String v = f.apply(e);
            if (v != null) e.details.put(field, v);
        }
    }

    public void removeVertexPreservingEdges(Vertex v) {
        if (vertexNodes.containsKey(v)) {
            BasicGraphNode n = vertexNodes.remove(v);
            invertedVertexNodes.remove(n);
            Set<BasicGraphEdge> incommingEdges = findEdgesTo(n);
            Set<BasicGraphEdge> outgoingEdges = findEdgesFrom(n);
            edges.removeAll(incommingEdges);
            edges.removeAll(outgoingEdges);
            // generate the cartesian product
            for (BasicGraphEdge i : incommingEdges) {
                for (BasicGraphEdge o : outgoingEdges) {
                    mergeEdge(i, o);
                }
            }
        }
    }

    private void mergeEdge(BasicGraphEdge i, BasicGraphEdge o) {
        BasicGraphEdge e = new BasicGraphEdge(i.getSource(), o.getDestination());
        mergeMetadata(e.details, i.details, o.details);
        edges.add(e);
    }

    @Override
    public Collection<BasicGraphNode> getNodes() {
        return vertexNodes.values();
    }

    @Override
    public Collection<BasicGraphEdge> getEdges() {
        return edges;
    }
}
