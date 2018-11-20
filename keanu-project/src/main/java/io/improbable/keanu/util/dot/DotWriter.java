package io.improbable.keanu.util.dot;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkWriter;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.SaveParentVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;


/**
 * Utility class for outputting a network to a DOT file that can then be used by a range of graph visualisers.
 * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
 *
 * Usage:
 * To output network to a DOT file: DotWriter.outputDot(fileName, network)
 * To output vertex and its connections up to degree n: DotWriter.outputDot(fileName, vertex, n)
 */
public class DotWriter implements NetworkWriter{

    private static final String DOT_HEADER = "digraph BayesianNetwork {\n";
    private static final String DOT_ENDING = "}";
    private static final int INFINITE_NETWORK_DEGREE = Integer.MAX_VALUE;

    private Set<VertexDotLabel> dotLabels = new HashSet<>();
    private Set<GraphEdge> graphEdges = new HashSet<>();
    private BayesianNetwork bayesianNetwork;

    public DotWriter(BayesianNetwork network) {
        bayesianNetwork = network;
    }

    // TODO clean save values everywhere, or would we want it?
    /**
     * Outputs the network to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param output output stream to use for writing
     * @param saveValues specify whether you want to output values of non-constant scalar vertices
     */
    @Override
    public void save(OutputStream output, boolean saveValues) throws IOException {
        Preconditions.checkArgument(bayesianNetwork.getAllVertices().size() > 0, "Network must contain at least one vertex.");
        Vertex anyVertex = bayesianNetwork.getAllVertices().get(0);
        save(output, anyVertex, INFINITE_NETWORK_DEGREE, saveValues);
    }

    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param output output stream to use for writing
     * @param vertex vertex around which the graph will be visualised
     * @param degree degree of connections to be visualised; for instance, if the degree is 1,
     *               only connections between the vertex and its parents and children will be written out to the DOT file.
     * @param saveValues specify whether you want to output values of non-constant scalar vertices
     */
    public void save(OutputStream output, Vertex vertex, int degree, boolean saveValues) throws IOException {

        dotLabels = new HashSet<>();
        graphEdges = new HashSet<>();

        Writer outputWriter = new OutputStreamWriter(output);

        Set<Vertex> subGraph = bayesianNetwork.getSubgraph(vertex, degree);
        for (Vertex v : subGraph) {
            if (saveValues) {
                v.saveValue(this);
            }
            else {
                v.save(this);
            }
        }

        outputWriter.write(DOT_HEADER);
        outputEdges(graphEdges, outputWriter, subGraph);
        outputLabels(dotLabels, outputWriter);
        outputWriter.write(DOT_ENDING);
        outputWriter.close();
    }

    private static void outputLabels(Collection<VertexDotLabel> dotLabels, Writer outputWriter) throws IOException {
        for (VertexDotLabel dotLabel: dotLabels) {
            outputWriter.write(dotLabel.inDotFormat() + "\n");
        }
    }

    private static void outputEdges(Collection<GraphEdge> edges, Writer outputWriter, Set<Vertex> verticesToOutput) throws IOException {
        for (GraphEdge edge : edges) {
            if (verticesToOutput.contains(edge.getParentVertex()) && verticesToOutput.contains(edge.getChildVertex())) {
                outputWriter.write(EdgeDotLabel.inDotFormat(edge) + "\n");
            }
        }
    }

    @Override
    public void save(Vertex vertex) {
        dotLabels.add(getDotLabel(vertex));
        graphEdges.addAll(getParentEdges(vertex));
    }

    @Override
    public void save(ConstantVertex vertex) {
        saveValue((Vertex)vertex);
    }

    @Override
    public void saveValue(Vertex vertex) {
        if (vertex.hasValue() && vertex.getValue() instanceof Tensor) {
            setDotLabelWithValue(vertex);
        } else {
            dotLabels.add(getDotLabel(vertex));
        }
    }

    @Override
    public void saveValue(DoubleVertex vertex) {
        setDotLabelWithValue(vertex);
        graphEdges.addAll(getParentEdges(vertex));
    }

    @Override
    public void saveValue(IntegerVertex vertex) {
        setDotLabelWithValue(vertex);
        graphEdges.addAll(getParentEdges(vertex));
    }

    @Override
    public void saveValue(BoolVertex vertex) {
        setDotLabelWithValue(vertex);
        graphEdges.addAll(getParentEdges(vertex));
    }

    private void setDotLabelWithValue(Vertex<? extends Tensor> vertex) {
        VertexDotLabel vertexDotLabel = getDotLabel(vertex);
        if (vertex.hasValue() && vertex.getValue().isScalar()) {
            vertexDotLabel.setValue("" + vertex.getValue().scalar());
        }
        dotLabels.add(vertexDotLabel);
    }

    private VertexDotLabel getDotLabel(Vertex vertex) {
        VertexDotLabel vertexDotLabel = new VertexDotLabel(vertex);
        if (vertex.getLabel() != null) {
            vertexDotLabel.setVertexLabel(vertex.getLabel().getUnqualifiedName());
        }
        DisplayInformationForOutput vertexAnnotation = vertex.getClass().getAnnotation(DisplayInformationForOutput.class);
        if (vertexAnnotation != null && !vertexAnnotation.displayName().isEmpty()) {
            vertexDotLabel.setAnnotation(vertexAnnotation.displayName());
        }
        return vertexDotLabel;
    }

    private Set<GraphEdge> getParentEdges(Vertex vertex) {
        Set<GraphEdge> edges = new HashSet<>();
        for (Object v : vertex.getParents()) {
            edges.add(new GraphEdge((Vertex)v, vertex));
        }

        // Check if any of the edges represent a connection between the vertex and its hyperparameter and annotate it accordingly.
        Class vertexClass = vertex.getClass();
        Method[] methods = vertexClass.getMethods();

        for (Method method : methods) {
            SaveParentVertex annotation = method.getAnnotation(SaveParentVertex.class);
            if (annotation != null) {
                String parentName = annotation.value();
                try {
                    Vertex parentVertex = (Vertex)method.invoke(vertex);
                    GraphEdge parentEdge = new GraphEdge(vertex, parentVertex);
                    edges.stream().filter(parentEdge::equals).findFirst().get().appendToLabel(parentName);
                } catch (Exception e) {
                    throw new IllegalArgumentException("Invalid parent retrieval function specified", e);
                }
            }
        }
        return edges;
    }
}
