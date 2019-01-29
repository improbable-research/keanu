package io.improbable.keanu.util.io;

import io.improbable.keanu.annotation.DisplayInformationForOutput;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;

import java.util.*;

public class DefaultDecorator implements DotDecorator {

    @Override
    public boolean includeVertex(Vertex v) {
        return true;
    }

    @Override
    public boolean includeEdge(GraphEdge edge) {
        return includeVertex(edge.getChildVertex()) && includeVertex(edge.getParentVertex());
    }

    @Override
    public Collection<String> labelEdge(GraphEdge edge) {
        return edge.getLabels();
    }

    @Override
    public Map<String, String> getExtraEdgeFields(GraphEdge edge) {
        return new HashMap<>();
    }

    @Override
    public Collection<String> labelVertex(Vertex v) {
        List<String> labels = new LinkedList<>();
        labels.add(getDescriptiveInfo(v));
        return labels;
    }

    private String getDescriptiveInfo(Vertex v) {
        if (v.getLabel() != null) {
            return v.getLabel().toString();
        }
        DisplayInformationForOutput vertexAnnotation = v.getClass().getAnnotation(DisplayInformationForOutput.class);
        if (vertexAnnotation != null) {
            return vertexAnnotation.displayName();
        }
        return v.getClass().getSimpleName();
    }

    @Override
    public Map<String, String> getExtraVertexFields(Vertex v) {
        return new HashMap<>();
    }
}
