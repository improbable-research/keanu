package io.improbable.keanu.util.graph.io;

import io.improbable.keanu.util.graph.AbstractGraph;
import io.improbable.keanu.util.graph.GraphEdge;
import io.improbable.keanu.util.graph.GraphNode;

import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Map;

public abstract class GraphToDot {

    private static final String DOT_HEADER = "digraph BayesianNetwork {\n";
    private static final String DOT_ENDING = "}";
    private static final String DOT_COMMENT_APPENDIX = "// ";
    private static final String DOT_EDGE = " -> ";

    private static final String DOT_FIELD_OPENING = " [";
    private static final String DOT_FIELD_SEPARATOR = "=\"";
    private static final String DOT_FIELD_CLOSING = "\"]";
    private static final String DOT_NEW_LINE = "\n";

    public static void write(AbstractGraph<? extends GraphNode, ? extends GraphEdge> graph, OutputStream output) throws IOException {
        Writer outputWriter = new OutputStreamWriter(output);
        write(graph, outputWriter);
    }

    public static void write(AbstractGraph<? extends GraphNode, ? extends GraphEdge> graph, Writer outputWriter) throws IOException {
        outputWriter.write(DOT_HEADER);
        for (Map.Entry<String, String> e : graph.getMetaData().entrySet()) {
            outputWriter.write(DOT_COMMENT_APPENDIX);
            outputWriter.write(e.getKey());
            outputWriter.write(e.getValue());
            outputWriter.write(DOT_NEW_LINE);
        }
        for (GraphNode n : graph.getNodes()) {
            write(n, outputWriter);
        }
        for (GraphEdge e : graph.getEdges()) {
            write(e, outputWriter);
        }
        outputWriter.write(DOT_ENDING);
        outputWriter.close();
    }

    protected static void write(GraphNode n, Writer outputWriter) throws IOException {
        outputWriter.write(Integer.toString(n.getIndex()));
        write(n.getDetails(), outputWriter);
        outputWriter.write(DOT_NEW_LINE);
    }

    protected static void write(GraphEdge e, Writer outputWriter) throws IOException {
        outputWriter.write(Integer.toString(e.getSource().getIndex()));
        outputWriter.write(DOT_EDGE);
        outputWriter.write(Integer.toString(e.getDestination().getIndex()));
        write(e.getDetails(), outputWriter);
        outputWriter.write(DOT_NEW_LINE);
    }

    protected static void write(Map<String, String> d, Writer outputWriter) throws IOException {
        for (Map.Entry<String, String> e : d.entrySet()) {
            outputWriter.write(DOT_FIELD_OPENING);
            outputWriter.write(e.getKey());
            outputWriter.write(DOT_FIELD_SEPARATOR);
            outputWriter.write(e.getValue());
            outputWriter.write(DOT_FIELD_CLOSING);
        }
    }
}
