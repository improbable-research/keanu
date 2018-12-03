package io.improbable.keanu.util.dot;

import com.google.common.base.Preconditions;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
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
 * <p>
 * Usage:
 * Create dotSaver: DotSaver writer = new DotSaver(yourBayesianNetwork);
 * To output network to a DOT file: writer.save(outputStream, saveValues);
 * To output vertex and its connections up to degree n: writer.save(outputStream, vertex, degree, saveValues);
 * where saveValues specifies whether you want to output values for vertices for which they've been set.
 */
public class DotSaver implements NetworkSaver {

    private static final String DOT_HEADER = "digraph BayesianNetwork {\n";
    private static final String DOT_ENDING = "}";
    private static final int INFINITE_NETWORK_DEGREE = Integer.MAX_VALUE;

    private Set<VertexDotLabel> dotLabels = new HashSet<>();
    private Set<GraphEdge> graphEdges = new HashSet<>();
    private BayesianNetwork bayesianNetwork;

    public DotSaver(BayesianNetwork network) {
        bayesianNetwork = network;
    }

    /**
     * Outputs the network to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param output     output stream to use for writing
     * @param saveValues specify whether you want to output values of non-constant scalar vertices
     * @throws IOException Any errors that occur during saving to the output stream
     */
    @Override
    public void save(OutputStream output, boolean saveValues) throws IOException {
        Preconditions.checkArgument(bayesianNetwork.getAllVertices().size() > 0, "Network must contain at least one vertex.");
        Vertex anyVertex = bayesianNetwork.getAllVertices().get(0);
        save(output, anyVertex, INFINITE_NETWORK_DEGREE, saveValues);
    }

    /**
     * Outputs a subgraph around the specified vertex to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param output     output stream to use for writing
     * @param vertex     vertex around which the subgraph will be centered
     * @param degree     degree of connections to be visualised; for instance, if the degree is 1,
     *                   only connections between the vertex and its parents and children will be written out to the DOT file.
     * @param saveValues specify whether you want to output values of non-constant scalar vertices
     * @throws IOException Any errors that occur during saving to the output stream
     */
    public void save(OutputStream output, Vertex vertex, int degree, boolean saveValues) throws IOException {

        dotLabels = new HashSet<>();
        graphEdges = new HashSet<>();
        Writer outputWriter = new OutputStreamWriter(output);

        Set<Vertex> subGraph = bayesianNetwork.getSubgraph(vertex, degree);
        for (Vertex v : subGraph) {
            if (saveValues) {
                v.saveValue(this);
            } else {
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
        for (VertexDotLabel dotLabel : dotLabels) {
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
        dotLabels.add(new VertexDotLabel(vertex));
        graphEdges.addAll(getParentEdges(vertex));
    }

    @Override
    public void save(ConstantVertex vertex) {
        saveValue((Vertex) vertex);
    }

    @Override
    public void saveValue(Vertex vertex) {
        if (vertex.hasValue() && vertex.getValue() instanceof Tensor) {
            setDotLabelWithValue(vertex);
        } else {
            dotLabels.add(new VertexDotLabel(vertex));
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
        VertexDotLabel vertexDotLabel = new VertexDotLabel(vertex);
        if (vertex.hasValue() && vertex.getValue().isScalar()) {
            vertexDotLabel.setValue("" + vertex.getValue().scalar());
        }
        dotLabels.add(vertexDotLabel);
    }

    private Set<GraphEdge> getParentEdges(Vertex vertex) {
        Set<GraphEdge> edges = new HashSet<>();
        for (Object v : vertex.getParents()) {
            edges.add(new GraphEdge((Vertex) v, vertex));
        }

        // Check if any of the edges represent a connection between the vertex and its hyperparameter and annotate it accordingly.
        Class vertexClass = vertex.getClass();
        Method[] methods = vertexClass.getMethods();

        for (Method method : methods) {
            SaveVertexParam annotation = method.getAnnotation(SaveVertexParam.class);
            if (annotation != null && Vertex.class.isAssignableFrom(method.getReturnType())) {
                String parentName = annotation.value();
                try {
                    Vertex parentVertex = (Vertex) method.invoke(vertex);
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
