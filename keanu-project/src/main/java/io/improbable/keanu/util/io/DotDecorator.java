package io.improbable.keanu.util.io;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.Map;

public interface DotDecorator {

    boolean includeVertex( Vertex v );
    boolean includeEdge(GraphEdge edge);
    Collection<String> labelEdge(GraphEdge edge);
    Map<String, String> getExtraEdgeFields(GraphEdge edge);
    Collection<String> labelVertex( Vertex v );
    Map<String, String> getExtraVertexFields( Vertex v );
}
