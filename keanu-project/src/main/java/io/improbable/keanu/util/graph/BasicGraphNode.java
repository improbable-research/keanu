package io.improbable.keanu.util.graph;

import java.util.HashMap;
import java.util.Map;

public class BasicGraphNode implements GraphNode {

    public Map<String,String> details = new HashMap<>();
    public final int index;

    public BasicGraphNode(int index ){
        this.index = index;
    }

    @Override
    public Map<String, String> getDetails() {
        return details;
    }

    @Override
    public int getIndex() {
        return index;
    }
}
