package io.improbable.keanu.util.graph;

import io.improbable.keanu.vertices.VertexId;

import java.util.HashMap;
import java.util.Map;

public class BasicGraphNode implements GraphNode {

    public Map<String,String> details = new HashMap<>();
    public final long index;

    public BasicGraphNode(){
        this.index = VertexId.ID_GENERATOR.getAndIncrement();
    }

    @Override
    public Map<String, String> getDetails() {
        return details;
    }

    @Override
    public long getIndex() {
        return index;
    }
}
