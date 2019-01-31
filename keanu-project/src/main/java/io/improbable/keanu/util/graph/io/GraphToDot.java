package io.improbable.keanu.util.graph.io;

import io.improbable.keanu.util.graph.AbstractGraph;
import io.improbable.keanu.util.graph.GraphEdge;
import io.improbable.keanu.util.graph.GraphNode;
import io.improbable.keanu.util.graph.VertexGraph;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.util.Map;

public abstract class GraphToDot {

    private static final String DOT_HEADER = "digraph BayesianNetwork {\n";
    private static final String DOT_ENDING = "}";
    private static final String DOT_COMMENT_APPENDIX = "// ";
    private static final String DOT_EDGE = " -> ";

    private static final String DOT_FIELD_OPENING = " [";
    private static final String DOT_FIELD_ASSIGN = "=\"";
    private static final String DOT_FIELD_SEPARATOR = ",";
    private static final String DOT_FIELD_CLOSE = "\"";
    private static final String DOT_FIELD_CLOSING = "]";
    private static final String DOT_NEW_LINE = "\n";
    private static final String DOT_METADATA_SPACER = ": ";

    public static void writeFile(@NotNull AbstractGraph<? extends GraphNode, ? extends GraphEdge>  graph, @NotNull File file) throws IOException {
        FileOutputStream outputStream = new FileOutputStream(file);
        try {
            write(graph, outputStream);
        }finally {
            outputStream.close();
        }
    }

    public static void write(@NotNull AbstractGraph<? extends GraphNode, ? extends GraphEdge> graph, @NotNull OutputStream output) throws IOException {
        Writer outputWriter = new OutputStreamWriter(output);
        write(graph, outputWriter);
    }

    public static void write(@NotNull AbstractGraph<? extends GraphNode, ? extends GraphEdge> graph, @NotNull Writer outputWriter) throws IOException {
        graph.prepareForExport();
        outputWriter.write(DOT_HEADER);
        for (Map.Entry<String, String> e : graph.getMetaData().entrySet()) {
            outputWriter.write(DOT_COMMENT_APPENDIX);
            outputWriter.write(e.getKey());
            outputWriter.write(DOT_METADATA_SPACER);
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

    protected static void write(@NotNull GraphNode n, @NotNull Writer outputWriter) throws IOException {
        outputWriter.write(Long.toString(n.getIndex()));
        write(n.getDetails(), outputWriter);
        outputWriter.write(DOT_NEW_LINE);
    }

    protected static void write(@NotNull GraphEdge e, @NotNull Writer outputWriter) throws IOException {
        outputWriter.write(Long.toString(e.getSource().getIndex()));
        outputWriter.write(DOT_EDGE);
        outputWriter.write(Long.toString(e.getDestination().getIndex()));
        write(e.getDetails(), outputWriter);
        outputWriter.write(DOT_NEW_LINE);
    }

    protected static void write(@NotNull Map<String, String> d, @NotNull Writer outputWriter) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append(DOT_FIELD_OPENING);
        boolean first = true;
        for (Map.Entry<String, String> e : d.entrySet()) {
            if ( first ){
                first = false;
            }else{
                sb.append(DOT_FIELD_SEPARATOR);
            }
            sb.append(makeSafe(e.getKey()));
            sb.append(DOT_FIELD_ASSIGN);
            sb.append(makeSafe(e.getValue()));
            sb.append(DOT_FIELD_CLOSE);
        }
        sb.append(DOT_FIELD_CLOSING);
        outputWriter.write(sb.toString());
    }

    private static String makeSafe(String input) {
        return input.replaceAll("[\n\"]" , "" );
    }
}
