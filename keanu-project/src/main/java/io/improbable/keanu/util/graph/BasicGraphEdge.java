package io.improbable.keanu.util.graph;

import java.util.HashMap;
import java.util.Map;

public class BasicGraphEdge implements GraphEdge<BasicGraphNode> {

    public Map<String,String> details = new HashMap<>();
    private BasicGraphNode source;
    private BasicGraphNode destination;


    public BasicGraphEdge( BasicGraphNode s , BasicGraphNode d ){
        source = s;
        destination = d;
    }

    @Override
    public BasicGraphNode getSource() {
        return source;
    }

    @Override
    public BasicGraphNode getDestination() {
        return destination;
    }

    @Override
    public Map<String, String> getDetails() {
        return details;
    }
}
