package io.improbable.keanu.util.graph;

import java.util.Map;

public interface GraphEdge<N extends GraphNode>{
    N getSource();
    N getDestination();
    Map<String,String> getDetails();
}
