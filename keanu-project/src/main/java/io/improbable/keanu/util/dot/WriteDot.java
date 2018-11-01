package io.improbable.keanu.util.dot;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.apache.commons.io.FilenameUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.stream.Stream;

/**
 * Utility class for outputting a network to a DOT file.
 * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
 */
public class WriteDot {

    private static BufferedWriter writer;
    private static HashMap<Integer, String> idsToLabelStrings = new HashMap<>();
    private static HashMap<String, String> verticeIdsToConnectionInfo = new HashMap<>();


    /**
     * Annotation used for vertex classes to specify how they should be exported to a DOT file.
     */
    @Retention(RetentionPolicy.RUNTIME)
    public @interface DotAnnotation {
        String displayName() default "";
        boolean displayHyperparameterInfo() default false;
    }


    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     *
     * @param fileName name of the output file.
     * @param net network to be written out to DOT format.
     */
    public static void outputDot(String fileName, BayesianNetwork net) {
        outputDot(fileName, "BayesianNetwork" ,net);
    }


    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param fileName name of the output file.
     * @param graphName mame of the graph (default - BayesianNetwork).
     * @param net network to be written out to DOT format.
     */
    public static void outputDot(String fileName, String graphName, BayesianNetwork net) {

        // Get any vertex in the network and set the degree to infinity to print out the entire network.
        Vertex someVertex = net.getTopLevelLatentOrObservedVertices().get(0);
        outputDot(fileName, graphName, someVertex, Integer.MAX_VALUE);
    }


    /**
     * Outputs the given graph to a DOT file which can be used by various graph visualisers to generate a visual representation of the graph.
     * Read more about DOT format here: https://en.wikipedia.org/wiki/DOT_(graph_description_language)
     *
     * @param fileName name of the output file
     * @param graphName mame of the graph (default - BayesianNetwork)
     * @param vertex vertex around which the graph will be visualised
     * @param degree degree of connections to be visualised; for instance, if the degree is 1,
     *               only connections between the vertex amd it's parents and children will be written out to the DOT file.
     */
    public static void outputDot(String fileName, String graphName, Vertex vertex, int degree) {
        try {
            // Reset variables
            idsToLabelStrings = new HashMap<>();
            verticeIdsToConnectionInfo = new HashMap<>();

            File outputFile = getOutputFile(fileName);

            writer = new BufferedWriter(new FileWriter(outputFile));
            writer.write("digraph " + graphName + " {\n");

            // Iterate over vertices and obtain the necessary label and conneciton info.
            obtainGraphInfo(vertex, degree);

            // Write out labels and connections.
            outputConnections(verticeIdsToConnectionInfo.values());
            outputLabels();

            writer.write("}");
            writer.close();
        }
        catch (IOException e) {
            System.out.println("Encountered an issue writing to the specified file (" + fileName + "):");
            e.printStackTrace();
        }
    }


    // Make sure the output directory exists and that no files are being overwritten.
    private static File getOutputFile(String fileName) {

        File outputFile = new File(fileName);

        // Make sure the output directory exists.
        if (!outputFile.getParentFile().exists()){
            outputFile.getParentFile().mkdirs();
        }

        // Add an index to the filename if a file with the specified name already exists.
        if (outputFile.exists()) {
            String baseName = FilenameUtils.getBaseName(fileName);
            String extension = FilenameUtils.getExtension(fileName);
            int counter = 1;
            while(outputFile.exists())
            {
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
    private static void obtainGraphInfo(Vertex vertex, int degree) {

        HashSet<VertexId> processedVertices = new HashSet<>();
        HashSet<Vertex> verticesToProcessNow = new HashSet<>();
        verticesToProcessNow.add(vertex);

        // Iterate over the graph till the specified degree or till there are no more unprocessed vertices left.
        int iterationIndex = 0;
        while(iterationIndex < degree && !verticesToProcessNow.isEmpty()) {

            HashSet<Vertex> verticesToProcessNext = new HashSet<>();

            for (Vertex v : verticesToProcessNow) {

                // Use hashed IDs as the unique identifier for each vertex and store the other information in labels.
                VertexId vId = v.getId();
                setLabel(v);

                // Iterate over all vertices connected to this one.
                Stream<Vertex> connectedVertices = Stream.concat(v.getParents().stream(), v.getChildren().stream());
                connectedVertices.forEach(connectedVertex -> {
                    VertexId connectedVId = connectedVertex.getId();
                    if (!processedVertices.contains(connectedVId)) {
                        verticeIdsToConnectionInfo.put(concatenateVertexIds(connectedVId, vId), getEdgeString(connectedVId, vId));
                        verticesToProcessNext.add(connectedVertex);
                    }
                });

                // If there is any hyperparameter information to be added to edges, add it now.
                DotAnnotation vertexAnnotation = v.getClass().getAnnotation(DotAnnotation.class);
                if (vertexAnnotation != null && vertexAnnotation.displayHyperparameterInfo()) {
                    applyHyperparameterInfo(v);
                }

                processedVertices.add(vId);
            }

            verticesToProcessNow = verticesToProcessNext;
            iterationIndex++;
        }

        // Set labels for vertices that have not yet been processed.
        for (Vertex v : verticesToProcessNow) {
            setLabel(v);
        }
    }


    // Utility function for creating a unique identifier for a connection between two vertices.
    private static String concatenateVertexIds(VertexId v1, VertexId v2) {
        if (v1.compareTo(v2) < 0) {
            return v1.toString() + "_" + v2.toString();
        }
        else {
            return v2.toString() + "_" + v1.toString();
        }
    }


    // Utility function for forming a DOT string to represent an edge.
    private static String getEdgeString(VertexId v1, VertexId v2) {
        if (v1.compareTo(v2) < 0) {
            return  "<" + v1.hashCode() + "> -> <" + v2.hashCode() + ">";
        }
        else {
            return  "<" + v2.hashCode() + "> -> <" + v1.hashCode() + ">";
        }
    }


    /**
     * Creates and stores a label for the given vertex.
     * Label contains the information that will be displayed for this vertex, such as vertex class or display name and value.
     *
     * @param v a vertex
     */
    private static void setLabel(Vertex v){

        String vertexLabel;

        // If vertex is a constant vertex, display only it's value.
        if ((vertexLabel = getValueAsString(v)).equals("")) {
            // Else use vertex label as DOT label for it.
            DotAnnotation vertexAnnotation = v.getClass().getAnnotation(DotAnnotation.class);
            if (v.getLabel() != null) {
                vertexLabel = v.getLabel().getUnqualifiedName();
            }
            // If label isn't set, and if DOT annotation is added to this class, use display name for label.
            else if (vertexAnnotation != null && !vertexAnnotation.displayName().equals("")) {
                vertexLabel = vertexAnnotation.displayName();
            }
            // Else use class name.
            else {
                vertexLabel = v.getClass().getSimpleName();
            }
        }

        int uniqueHiddenID = v.getId().hashCode();
        String fullName = uniqueHiddenID + "[label=\"" + vertexLabel + "\"]";
        idsToLabelStrings.put(uniqueHiddenID, fullName);
    }


    // Get value of constant vertices that always have their value set.
    private static String getValueAsString(Vertex v) {
        if (v.getClass().equals(ConstantDoubleVertex.class)) {
            return "" + ((ConstantDoubleVertex) v).getValue().getValue(0);
        }
        if (v.getClass().equals(ConstantIntegerVertex.class)) {
            return "" + ((ConstantIntegerVertex) v).getValue().getValue(0);
        }
        if (v.getClass().equals(ConstantBoolVertex.class)) {
            return "" + ((ConstantBoolVertex) v).getValue().getValue(0);
        }
        return "";
    }


    // Apply hyperparameters as a label to connection edges.
    // Note - this is currently only done for Gaussian vertices. Can be extended for any others.
    private static void applyHyperparameterInfo(Vertex v) {
        VertexId vId = v.getId();

        if (v.getClass().getSimpleName().equals(GaussianVertex.class.getSimpleName())) {
            VertexId meanVertexId = ((GaussianVertex) v).getMu().getId();
            verticeIdsToConnectionInfo.put(concatenateVertexIds(vId, meanVertexId), verticeIdsToConnectionInfo.get(concatenateVertexIds(vId, meanVertexId)) + " [label=mu]");
            VertexId sigmaVertexId = ((GaussianVertex) v).getSigma().getId();
            verticeIdsToConnectionInfo.put(concatenateVertexIds(vId, sigmaVertexId), verticeIdsToConnectionInfo.get(concatenateVertexIds(vId, sigmaVertexId)) + " [label=sigma]");
        }
    }


    // Output information about edges.
    private static void outputConnections(Collection<String> connections) throws IOException {
        for (String connection: connections) {
            writer.write(connection + "\n");
        }
    }


    // Output the stored labels.
    private static void outputLabels() throws IOException {
        for (String labelString: idsToLabelStrings.values()) {
            writer.write(labelString + "\n");
        }
    }


    public static void main(String[] args) {
        writerTest();
    }

    public static void writerTest() {
        Collection<? extends Vertex> vertices;

        double mu = 0;
        double sigma = 1;
//        GaussianVertex v1 = new GaussianVertex(mu, sigma);
//        ConstantIntegerVertex v1 = new ConstantIntegerVertex(4);
//        ConstantIntegerVertex v2 = new ConstantIntegerVertex(5);
//        ConstantBoolVertex v1 = new ConstantBoolVertex(true);
//        ConstantBoolVertex v2 = new ConstantBoolVertex(false);
//
//        IntegerVertex v4 = v1.multiply(v2);
//
//        BayesianNetwork myNet = new BayesianNetwork(v4.getConnectedGraph());

        GaussianVertex v1 = new GaussianVertex(mu, sigma);
        ConstantDoubleVertex v2 = new ConstantDoubleVertex(5);

        DoubleVertex v4 = v2.multiply(v1);

        BayesianNetwork myNet = new BayesianNetwork(v4.getConnectedGraph());

        outputDot("testdir/somemore/TestFile.txt", myNet);

//        outputDot("TestFile.txt", "VerticeConnections", v1, 2);

    }
}
