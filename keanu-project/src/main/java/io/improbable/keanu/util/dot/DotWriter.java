package io.improbable.keanu.util.dot;

import com.google.common.base.Preconditions;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.NetworkWriter;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;


/**
 * Utility class for outputting a network to a DOT file that can then be used by a range of graph visualisers.
 * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
 *
 * Usage:
 * To output network to a DOT file: DotWriter.outputDot(fileName, network)
 * To output vertex and it's connections up to degree n: DotWriter.outputDot(fileName, vertex, n)
 */
public class DotWriter implements NetworkWriter{

    private static final String DOT_HEADER = "digraph BayesianNetwork {\n";
    private static final String DOT_ENDING = "}";

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
        // Get any vertex in the network and set the degree to infinity to print out the entire network.
        Preconditions.checkArgument(bayesianNetwork.getAllVertices().size() > 0, "Network must contain at least one vertex.");
        Vertex someVertex = bayesianNetwork.getAllVertices().get(0);
        save(output, someVertex, Integer.MAX_VALUE, saveValues);
    }

    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param output output stream to use for writing
     * @param vertex vertex around which the graph will be visualised
     * @param degree degree of connections to be visualised; for instance, if the degree is 1,
     *               only connections between the vertex amd it's parents and children will be written out to the DOT file.
     * @param saveValues specify whether you want to output values of non-constant scalar vertices
     */
    @Override
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

    // Output information about labels.
    private static void outputLabels(Collection<VertexDotLabel> dotLabels, Writer outputWriter) throws IOException {
        for (VertexDotLabel dotLabel: dotLabels) {
            outputWriter.write(dotLabel.inDotFormat() + "\n");
        }
    }

    // Output information about edges.
    private static void outputEdges(Collection<GraphEdge> edges, Writer outputWriter, Set<Vertex> verticesToOutput) throws IOException {
        for (GraphEdge edge : edges) {
            // Only output edge if both of the vertices it connects will be written out.
            if (verticesToOutput.contains(edge.getParentVertex()) && verticesToOutput.contains(edge.getChildVertex())) {
                outputWriter.write(edge.inDotFormat() + "\n");
            }
        }
    }

    @Override
    public void save(Vertex vertex) {
        dotLabels.add(vertex.getDotLabel());
        graphEdges.addAll(vertex.getParentEdgesInDotFormat());
    }

    @Override
    public void save(ConstantVertex vertex) {
        if (vertex instanceof ConstantVertex) {
            saveValue((Vertex)vertex);
            return;
        }
    }

    @Override
    public void saveValue(Vertex vertex) {
        setDotLabelWithValue(vertex);
        graphEdges.addAll(vertex.getParentEdgesInDotFormat());
    }

    @Override
    public void saveValue(DoubleVertex vertex) {
        setDotLabelWithValue(vertex);
        graphEdges.addAll(vertex.getParentEdgesInDotFormat());
    }

    @Override
    public void saveValue(IntegerVertex vertex) {
        setDotLabelWithValue(vertex);
        graphEdges.addAll(vertex.getParentEdgesInDotFormat());
    }

    @Override
    public void saveValue(BoolVertex vertex) {
        setDotLabelWithValue(vertex);
        graphEdges.addAll(vertex.getParentEdgesInDotFormat());
    }

    private void setDotLabelWithValue(Vertex<? extends Tensor> vertex) {
        VertexDotLabel vertexDotLabel = vertex.getDotLabel();
        if (vertex.getValue().isScalar()) {
            vertexDotLabel.setDotLabel(VertexDotLabel.VertexDotLabelType.VALUE, "" + vertex.getValue().scalar());
        }
        dotLabels.add(vertexDotLabel);
    }
}
