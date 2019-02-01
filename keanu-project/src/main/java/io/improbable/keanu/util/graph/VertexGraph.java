package io.improbable.keanu.util.graph;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.*;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.awt.*;
import java.lang.reflect.Method;
import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * This class implements AbstractGraph that can be generated from a BayesianNetwork
 */
public class VertexGraph extends AbstractGraph<BasicGraphNode, BasicGraphEdge> {

    private static final Color CONSTANT_COLOR = Color.BLACK;
    private static final Color PROBABILISTIC_COLOR = Color.PINK;
    private static final Color OBSERVED_COLOR = Color.BLUE;
    private static final Color DETERMINISTIC_COLOR = Color.CYAN;

    private static final String LABEL_KEY = "label";
    private static final String COLOR_KEY = "color";
    private static final Color DOUBLE_COLOR = Color.red;
    private static final Color INTEGER_COLOR = Color.green;
    private static final Color BOOLEAN_COLOR = Color.blue;
    private static final Color OTHER_COLOR = Color.darkGray;
    private boolean formattingApplied = false;

    private Map<Vertex, BasicGraphNode> vertexNodes = new HashMap<>();
    private Map<BasicGraphNode, Vertex> invertedVertexNodes = new HashMap<>();

    /**
     * @param inputNetwork BayesianNetwork to create a graph from
     */
    public VertexGraph(BayesianNetwork inputNetwork) {
        this(inputNetwork.getAllVertices());
    }

    /**
     * @param inputNetwork BayesianNetwork to create the graph from
     * @param v            target vertex in the graph
     * @param degree       degrees to include
     */
    public VertexGraph(BayesianNetwork inputNetwork, Vertex v, int degree) {
        this(inputNetwork.getSubgraph(v, degree));
    }

    /**
     * @param v Vertex to start from - includes all connected vertices
     */
    public VertexGraph(Vertex v) {
        this(v.getConnectedGraph());
    }

    /**
     * @param inputList Collection of vertices to build - will not expand this list.
     */
    public VertexGraph(Collection<Vertex> inputList) {
        // we loop twice first to create the nodes, then second time we know the other end should exist if it's in scope
        for (Vertex v : inputList) {
            addVertex(v);
        }
        for (Vertex v : inputList) {
            addVertexEdges(v);
        }
    }

    /**
     * @param key   metadata key
     * @param value metadata vaule
     * @return itself so it's chainable
     */
    public VertexGraph putMetadata(String key, String value) {
        metadata.put(key, value);
        return this;
    }

    /**
     * Automatically populates metadata about these verticies
     *
     * @return itself so it's chainable
     */
    public VertexGraph autoPutMetadata() {
        metadata.put("vertices", Integer.toString(vertexNodes.size()));
        metadata.put("edges", Integer.toString(edgeCount()));
        metadata.put("timestamp", new Date().toString());
        return this;
    }

    /**
     * Called before export - this applies default formatting if none has been specified
     */
    public void prepareForExport() {
        if (!formattingApplied) {
            labelEdgesWithParameters().colorVerticesByType().colorEdgesByParent().labelConstantVerticesWithValue();
        }
    }

    private void addVertex(Vertex v) {
        if (!vertexNodes.containsKey(v)) {
            BasicGraphNode n = createGraphNodeFor(v);
            vertexNodes.put(v, n);
            invertedVertexNodes.put(n, v);
        }
    }

    private BasicGraphNode createGraphNodeFor(Vertex v) {
        BasicGraphNode n = new BasicGraphNode(VertexId.ID_GENERATOR.getAndIncrement());
        n.details.put(LABEL_KEY, getBasicNameForVertex(v));
        return n;
    }

    private String getBasicNameForVertex(Vertex v) {
        if (v.getLabel() != null) {
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

    /**
     * This removes all Deterministic vertices from the graph - but preserves connectivity.
     *
     * @return itself so it's chainable
     */
    public VertexGraph removeDeterministicVertices() {
        removeVerticesPreservingEdges((v) -> !v.isProbabilistic());
        return this;
    }

    /**
     * This removes all intermediate vertices from the graph - but preserves connectivity.
     * Intermediate vertices are defined as:
     * * Deterministic
     * * Not constant
     * * Having children
     *
     * @return itself so it's chainable
     */
    public VertexGraph removeIntermediateVertices() {
        removeVerticesPreservingEdges((v) ->
            !v.isProbabilistic() && !(v instanceof ConstantVertex) && !v.getChildren().isEmpty());
        return this;
    }

    /**
     * This removes a specified namespace from the graph - but preserves connectivity.
     *
     * @param namespace namespace to remove
     * @return itself so it's chainable
     */
    public VertexGraph removeNamespace(String namespace) {
        removeVerticesPreservingEdges((v) -> vertexInNamespace(v, namespace));
        return this;
    }

    private boolean vertexInNamespace(Vertex v, String namespace) {
        VertexLabel label = v.getLabel();
        if (label == null) return true;
        return label.isInNamespace(namespace);
    }

    /**
     * This removes all verticies matching the predicate from the graph, and any edges connected to them
     *
     * @param f predicate of vertices to remove
     */
    public void removeVertices(Predicate<Vertex> f) {
        Set<Vertex> toRemove = new HashSet<>();
        for (Vertex v : vertexNodes.keySet()) {
            if (f.test(v)) {
                toRemove.add(v);
            }
        }
        for (Vertex v : toRemove) {
            removeVertex(v);
        }
    }

    /**
     * This removes all verticies matching the predicate from the graph, but preserves connectivity
     *
     * @param f predicate of vertices to remove
     */
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

    /**
     * Remove a vertex (and associated edges)
     *
     * @param v vertex to remove
     */
    public void removeVertex(Vertex v) {
        if (vertexNodes.containsKey(v)) {
            BasicGraphNode n = vertexNodes.remove(v);
            invertedVertexNodes.remove(n);
            Set<BasicGraphEdge> edgesToRemove = findEdgesUsing(n);
            edges.removeAll(edgesToRemove);
        }
    }

    /**
     * Remove a vertex (preserving connectivity of associated edges)
     *
     * @param v the vertex to remove
     */
    public void removeVertexPreservingEdges(Vertex v) {
        if (vertexNodes.containsKey(v)) {
            BasicGraphNode n = vertexNodes.remove(v);
            invertedVertexNodes.remove(n);
            Set<BasicGraphEdge> incommingEdges = findEdgesTo(n);
            Set<BasicGraphEdge> outgoingEdges = findEdgesFrom(n);
            edges.removeAll(incommingEdges);
            edges.removeAll(outgoingEdges);
            Set<BasicGraphEdge> newEdges = new HashSet<>();
            // generate the cartesian product
            for (BasicGraphEdge i : incommingEdges) {
                for (BasicGraphEdge o : outgoingEdges) {
                    newEdges.add(mergeEdge(i, o));
                }
            }
            removeDuplicatedEdges(newEdges);
        }
    }

    /**
     * This colours all vertex by state (Observer, Probabilistic, Deterministic or Constant)
     *
     * @return itself - so it's chainable
     */
    public VertexGraph colorVerticesByState() {
        formattingApplied = true;
        setVertexMetadata(COLOR_KEY, this::convertStateToColor);
        return this;
    }

    /**
     * This colours all vertex by data type
     *
     * @return itself - so it's chainable
     */
    public VertexGraph colorVerticesByType() {
        formattingApplied = true;
        setVertexMetadata(COLOR_KEY, this::convertTypeToColor);
        return this;
    }

    /**
     * This colours all edges by their source (parent) color
     *
     * @return itself - so it's chainable
     */
    public VertexGraph colorEdgesByParent() {
        formattingApplied = true;
        setEdgeMetadata(COLOR_KEY, (e) -> e.getSource().details.getOrDefault(COLOR_KEY, null));
        return this;
    }

    /**
     * This colours all vertex by namespace
     *
     * @return itself - so it's chainable
     */
    public VertexGraph colorVerticesByNamespace() {
        formattingApplied = true;
        ColorSequence<String> seq = new ColorSequence<>();
        setVertexMetadata(COLOR_KEY, (v) -> formatColorForDot(seq.getOrChoseColor(getVertexNamespace(v))));
        return this;
    }

    /**
     * This colours all vertex by a specified key
     *
     * @param key the key to use to chose the color
     * @return itself - so it's chainable
     */
    public VertexGraph colorVerticesByKey(String key) {
        formattingApplied = true;
        ColorSequence<String> seq = new ColorSequence<>();
        setVertexNodeMetadata(COLOR_KEY, (v) -> formatColorForDot(seq.getOrChoseColor(v.details.get(key))));
        return this;
    }

    private String getVertexNamespace(Vertex v) {
        VertexLabel label = v.getLabel();
        System.out.println("Processing label " + label);
        if (label == null) return null;
        return label.getOuterNamespace().orElse("none");
    }

    private String convertStateToColor(Vertex v) {
        Color c;
        if (v.isObserved()) {
            c = OBSERVED_COLOR;
        } else if (v instanceof Probabilistic) {
            c = PROBABILISTIC_COLOR;
        } else if (v instanceof ConstantVertex) {
            c = CONSTANT_COLOR;
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

    /**
     * This labels all vertex by their value
     *
     * @return itself - so it's chainable
     */
    public VertexGraph labelVerticesWithValue() {
        formattingApplied = true;
        setVertexMetadata(LABEL_KEY, this::convertValueToString);
        return this;
    }

    /**
     * This labels constant vertex by their value
     *
     * @return itself - so it's chainable
     */
    public VertexGraph labelConstantVerticesWithValue() {
        formattingApplied = true;
        setVertexMetadata(LABEL_KEY, this::convertConstantValueToString);
        return this;
    }

    /**
     * This labels all edges by their parameter name
     *
     * @return itself - so it's chainable
     */
    public VertexGraph labelEdgesWithParameters() {
        formattingApplied = true;
        setEdgeMetadata(LABEL_KEY, this::readEdgeName);
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

    private String convertConstantValueToString(Vertex vertex) {
        if (!(vertex instanceof ConstantVertex)) return null;
        return convertValueToString(vertex);
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

    /**
     * Sets a vertex metadata based on the Vertex
     *
     * @param key The key to set
     * @param f   The mapping to define the new value
     * @return itself - so it's chainable
     */
    public VertexGraph setVertexMetadata(String key, Function<Vertex, String> f) {
        formattingApplied = true;
        for (Map.Entry<Vertex, BasicGraphNode> e : vertexNodes.entrySet()) {
            String v = f.apply(e.getKey());
            if (v != null) e.getValue().details.put(key, v);
        }
        return this;
    }

    /**
     * Sets a vertex metadata based on the Node's existing metadata
     *
     * @param key The key to set
     * @param f   The mapping to define the new value
     * @return itself - so it's chainable
     */
    public VertexGraph setVertexNodeMetadata(String key, Function<BasicGraphNode, String> f) {
        formattingApplied = true;
        for (BasicGraphNode node : vertexNodes.values()) {
            String v = f.apply(node);
            if (v != null) node.details.put(key, v);
        }
        return this;
    }

    /**
     * Sets an edges metadata based on the Edge
     *
     * @param key the key to set
     * @param f   the mapping to define the new value
     * @return itself - so it's chainable
     */
    public VertexGraph setEdgeMetadata(String key, Function<BasicGraphEdge, String> f) {
        formattingApplied = true;
        for (BasicGraphEdge e : edges) {
            String v = f.apply(e);
            if (v != null) e.details.put(key, v);
        }
        return this;
    }

    private void removeDuplicatedEdges(Set<BasicGraphEdge> edgesToCheck) {
        Set<BasicGraphEdge> known = new HashSet<>();
        for (BasicGraphEdge e : edgesToCheck) {
            boolean found = false;
            for (BasicGraphEdge k : known) {
                if (e.getSource() == k.getSource() && e.getDestination() == k.getDestination()) {
                    found = true;
                    break;
                }
            }
            if (found) {
                edges.remove(e);
            } else {
                known.add(e);
            }
        }
    }

    /**
     * Updates any edge between two vertices
     *
     * @param a     first vertex
     * @param b     second vertex
     * @param key   key to update
     * @param value value to use
     * @return itself - so it's chainable
     * <p>
     * this will look for any edge between a and b (regardless of direction)
     */
    public VertexGraph putEdgeMetadataBetween(Vertex a, Vertex b, String key, String value) {
        BasicGraphNode nodeA = vertexNodes.get(a);
        BasicGraphNode nodeB = vertexNodes.get(b);
        if (nodeA == null || nodeB == null) return this;
        for (BasicGraphEdge edge : edges) {
            if ((edge.getSource() == nodeA && edge.getDestination() == nodeB) ||
                (edge.getSource() == nodeB && edge.getDestination() == nodeA)) {
                edge.details.put(key, value);
            }
        }
        return this;
    }

    /**
     * Updates the metadata for a specific vertex
     *
     * @param v     the vertex to update
     * @param key   the key to set
     * @param value the value to set to the key
     */
    public void putNodeMetadata(Vertex v, String key, String value) {
        if (vertexNodes.containsKey(v)) {
            vertexNodes.get(v).details.put(key, value);
        }
    }

    private BasicGraphEdge mergeEdge(BasicGraphEdge i, BasicGraphEdge o) {
        BasicGraphEdge e = new BasicGraphEdge(i.getSource(), o.getDestination(), o.getDetails());
        edges.add(e);
        return e;
    }

    @Override
    public Collection<BasicGraphNode> getNodes() {
        return vertexNodes.values();
    }

}
