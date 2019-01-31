package io.improbable.keanu.util.graph;

import java.util.HashMap;
import java.util.Map;

public class BasicGraphNode implements GraphNode {

    public final long index;
    public Map<String, String> details = new HashMap<>();

    public BasicGraphNode(long index) {
        this.index = index;
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
