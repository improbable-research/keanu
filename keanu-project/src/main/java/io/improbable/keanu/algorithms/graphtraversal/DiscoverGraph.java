package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;

import java.util.HashSet;
import java.util.Set;

public class DiscoverGraph {

    public static Set<Vertex<?>> getEntireGraph(Vertex<?> aVertex) {
        return getUpstream(aVertex, getDownstream(aVertex, new HashSet<>()));
    }

    private static Set<Vertex<?>> getDownstream(Vertex<?> aVertex, Set<Vertex<?>> discoveredGraph) {
        discoveredGraph.add(aVertex);

        for (Vertex<?> child : aVertex.getChildren()) {
            if (!discoveredGraph.contains(child)) {
                getDownstream(child, discoveredGraph);
                getUpstream(child, discoveredGraph);
            }
        }

        return discoveredGraph;
    }

    private static Set<Vertex<?>> getUpstream(Vertex<?> aVertex, Set<Vertex<?>> discoveredGraph) {
        discoveredGraph.add(aVertex);

        for (Vertex<?> parent : aVertex.getParents()) {
            if (!discoveredGraph.contains(parent)) {
                getUpstream(parent, discoveredGraph);
                getDownstream(parent, discoveredGraph);
            }
        }

        return discoveredGraph;
    }
}
