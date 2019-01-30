package io.improbable.keanu.util.graph;

import java.util.Map;

public interface GraphNode {
    Map<String,String> getDetails();

    long getIndex();
}
