package io.improbable.keanu.util.dot;

import com.google.common.base.Preconditions;
import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.apache.commons.io.FilenameUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * Utility class for outputting a network to a DOT file that can then be used by a range of graph visualisers.
 * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
 *
 * Usage:
 * To output network to a DOT file: WriteDot.outputDot(fileName, network)
 * To output vertex and it's connections up to degree n: WriteDot.outputDot(fileName, vertex, n)
 */
public class WriteDot {

    private static final String DOT_HEADER = "digraph BayesianNetwork {\n";
    private static final String DOT_ENDING = "}";

    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param fileName name of the output file.
     * @param net network to be written out to DOT format.
     */
    public static void outputDot(String fileName, BayesianNetwork net) {

        // Get any vertex in the network and set the degree to infinity to print out the entire network.
        Preconditions.checkArgument(net.getAllVertices().size() > 0, "Network must contain at least one vertex.");
        Vertex someVertex = net.getAllVertices().get(0);
        outputDot(fileName, someVertex, Integer.MAX_VALUE);
    }

    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param outputWriter stream to use for outputting the results
     * @param net network to be written out to DOT format.
     */
    public static void outputDot(Writer outputWriter, BayesianNetwork net) {

        // Get any vertex in the network and set the degree to infinity to print out the entire network.
        Preconditions.checkArgument(net.getAllVertices().size() > 0, "Network must contain at least one vertex.");
        Vertex someVertex = net.getAllVertices().get(0);
        outputDot(outputWriter, someVertex, Integer.MAX_VALUE);
    }

    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param fileName name of the output file
     * @param vertex vertex around which the graph will be visualised
     * @param degree degree of connections to be visualised; for instance, if the degree is 1,
     *               only connections between the vertex amd it's parents and children will be written out to the DOT file.
     */
    public static void outputDot(String fileName, Vertex vertex, int degree) {
        File outputFile = getOutputFile(fileName);

        try (Writer writer = new BufferedWriter(new FileWriter(outputFile))){
            outputDot(writer, vertex, degree);
        }
        catch (IOException e) {
            System.out.println("Encountered an issue creating the specified file (" + fileName + "):");
            e.printStackTrace();
        }
    }

    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param outputWriter stream to use for outputting the results
     * @param vertex vertex around which the graph will be visualised
     * @param degree degree of connections to be visualised; for instance, if the degree is 1,
     *               only connections between the vertex amd it's parents and children will be written out to the DOT file.
     */
    public static void outputDot(Writer outputWriter, Vertex vertex, int degree) {
        try {
            Set<String> dotLabels = new HashSet<>();
            Map<String, String> verticeIdsToConnectionInfo = new HashMap<>();

            outputWriter.write(DOT_HEADER);

            // Iterate over vertices and obtain the necessary label and connection info.
            obtainGraphInfo(vertex, degree, dotLabels, verticeIdsToConnectionInfo);

            // Write out labels and connections.
            outputInfo(verticeIdsToConnectionInfo.values(), outputWriter);
            outputInfo(dotLabels, outputWriter);

            outputWriter.write(DOT_ENDING);
            outputWriter.close();
        }
        catch (IOException e) {
            System.out.println("Encountered an issue when writing the output:");
            e.printStackTrace();
        }
    }

    // Make sure the output directory exists and that no files are being overwritten.
    private static File getOutputFile(String fileName) {

        File outputFile = getFreshOutputFile(fileName);

        // Make sure the output directory exists.
        if (outputFile.getParentFile() != null && !outputFile.getParentFile().exists()){
            outputFile.getParentFile().mkdirs();
        }

        return outputFile;
    }

    // Add an index to the filename if a file with the specified name already exists.
    private static File getFreshOutputFile(String fileName) {
        File outputFile = new File(fileName);

        if (outputFile.exists()) {
            String baseName = FilenameUtils.getBaseName(fileName);
            String extension = FilenameUtils.getExtension(fileName);
            int counter = 1;
            while(outputFile.exists()) {
                outputFile = new File(outputFile.getParent(), baseName + "_" + (counter++) + "." + extension);
            }
        }

        return outputFile;
    }

    /**
     * Iterates over the vertices of the graph, storing information about vertex labels to display and connections between vertices.
     *
     * @param vertex vertex around which the graph will be visualised
     * @param degree degree of connections to be visualised; for instance, if the degree is 1,
     *               only connections between the vertex amd it's parents and children will be written out to the DOT file.
     */
    private static void obtainGraphInfo(Vertex vertex, int degree, Set<String> dotLabels, Map<String, String> verticeIdsToConnectionInfo) {

        Set<VertexId> processedVertices = new HashSet<>();
        Set<Vertex> verticesToProcessNow = new HashSet<>();
        verticesToProcessNow.add(vertex);
        setDotLabel(vertex, dotLabels);

        // Iterate over the graph till the specified degree or till there are no more unprocessed vertices left.
        int iterationIndex = 0;
        while (iterationIndex < degree && !verticesToProcessNow.isEmpty()) {

            Set<Vertex> verticesToProcessNext = new HashSet<>();

            for (Vertex v : verticesToProcessNow) {

                // Use hashed IDs as the unique identifier for each vertex and store the other information in labels.
                VertexId vId = v.getId();

                // Iterate over all vertices connected to this one.
                Stream<Vertex> connectedVertices = Stream.concat(v.getParents().stream(), v.getChildren().stream());
                connectedVertices.forEach(connectedVertex -> {
                    VertexId connectedVId = connectedVertex.getId();
                    if (!processedVertices.contains(connectedVId)) {
                        verticeIdsToConnectionInfo.put(concatenateVertexIds(connectedVId, vId), getEdgeString(connectedVId, vId));
                        verticesToProcessNext.add(connectedVertex);
                        setDotLabel(connectedVertex, dotLabels);
                    }
                });

                // If there is any hyperparameter information to be added to edges, add it now.
                DisplayInformationForOutput vertexAnnotation = v.getClass().getAnnotation(DisplayInformationForOutput.class);
                if (vertexAnnotation != null && vertexAnnotation.displayHyperparameterInfo()) {
                    applyHyperparameterInfo(v, verticeIdsToConnectionInfo);
                }

                processedVertices.add(vId);
            }

            verticesToProcessNow = verticesToProcessNext;
            iterationIndex++;
        }

    }

    // Utility function for creating a unique identifier for a connection between two vertices.
    private static String concatenateVertexIds(VertexId v1, VertexId v2) {
        if (v1.compareTo(v2) < 0) {
            return v1.toString() + "_" + v2.toString();
        } else {
            return v2.toString() + "_" + v1.toString();
        }
    }


    // Utility function for forming a DOT string to represent an edge.
    private static String getEdgeString(VertexId v1, VertexId v2) {
        if (v1.compareTo(v2) < 0) {
            return  "<" + v1.hashCode() + "> -> <" + v2.hashCode() + ">";
        } else {
            return  "<" + v2.hashCode() + "> -> <" + v1.hashCode() + ">";
        }
    }

    /**
     * Creates and stores a label for the given vertex.
     * Label contains the information that will be displayed for this vertex, such as vertex class or display name and value.
     *
     * @param v a vertex
     */
    private static void setDotLabel(Vertex v, Set<String> dotLabels){

        String vertexLabel = getValueAsString(v);

        // If vertex is a constant vertex, display only it's value.
        if (vertexLabel.isEmpty()) {
            // Else use vertex label as DOT label for it.
            DisplayInformationForOutput vertexAnnotation = v.getClass().getAnnotation(DisplayInformationForOutput.class);
            if (v.getLabel() != null) {
                vertexLabel = v.getLabel().getUnqualifiedName();
            }
            // If label isn't set, and if DOT annotation is added to this class, use display name for label.
            else if (vertexAnnotation != null && !vertexAnnotation.displayName().isEmpty()) {
                vertexLabel = vertexAnnotation.displayName();
            }
            // Else use class name.
            else {
                vertexLabel = v.getClass().getSimpleName();
            }
        }

        int uniqueHiddenID = v.getId().hashCode();
        String fullName = uniqueHiddenID + "[label=\"" + vertexLabel + "\"]";
        dotLabels.add(fullName);
    }

    // Get value of constant vertices that always have their value set.
    private static String getValueAsString(Vertex v) {
        if ((v instanceof ConstantDoubleVertex || v instanceof ConstantBoolVertex || v instanceof ConstantIntegerVertex) && ((Vertex<Tensor>) v).getValue().isScalar()) {
            return "" + ((Vertex<Tensor>) v).getValue().scalar();
        }
        return "";
    }

    // Apply hyperparameters as a label to connection edges.
    // Note - this is currently only done for Gaussian vertices. Can be extended for any others.
    private static void applyHyperparameterInfo(Vertex v, Map<String, String> verticeIdsToConnectionInfo) {
        VertexId vId = v.getId();

        if (v instanceof GaussianVertex) {
            VertexId meanVertexId = ((GaussianVertex) v).getMu().getId();
            verticeIdsToConnectionInfo.computeIfPresent(concatenateVertexIds(vId, meanVertexId), (ids, previousValue) -> previousValue + " [label=mu]");
            VertexId sigmaVertexId = ((GaussianVertex) v).getSigma().getId();
            verticeIdsToConnectionInfo.computeIfPresent(concatenateVertexIds(vId, sigmaVertexId), (ids, previousValue) -> previousValue + " [label=sigma]");
        }
    }

    // Output information about edges.
    private static void outputInfo(Collection<String> infSet, Writer outputWriter) throws IOException {
        for (String info: infSet) {
            outputWriter.write(info + "\n");
        }
    }
}
