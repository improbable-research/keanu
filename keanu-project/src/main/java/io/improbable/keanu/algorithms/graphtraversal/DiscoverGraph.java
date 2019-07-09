package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.IVertex;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

public class DiscoverGraph {

    private DiscoverGraph() {
    }

    /**
     * This algorithm visits all vertices in a graph. It's memory
     * requirements is on the order of number of vertices in the graph
     * and compute requirements on the order of edges in the graph.
     * <p>
     * If the graph is very large (i.e. greater than 100k vertices), it will be
     * much faster to keep track of vertices as they are created than
     * to create the graph and then collect it with this method.
     *
     * @param initialVertex starting vertex for graph discovery
     * @return a set containing EVERY vertex in a graph that the
     * starting vertex is apart of.
     */
    public static Set<IVertex> getEntireGraph(IVertex<?> initialVertex) {

        Set<IVertex> discoveredGraph = new HashSet<>();

        Deque<IVertex<?>> stack = new ArrayDeque<>();

        discoveredGraph.add(initialVertex);
        stack.addFirst(initialVertex);

        while (!stack.isEmpty()) {

            IVertex<?> visiting = stack.removeFirst();

            for (IVertex<?> child : visiting.getChildren()) {
                if (!discoveredGraph.contains(child)) {
                    stack.addFirst(child);
                    discoveredGraph.add(child);
                }
            }

            for (IVertex<?> parent : visiting.getParents()) {
                if (!discoveredGraph.contains(parent)) {
                    stack.addFirst(parent);
                    discoveredGraph.add(parent);
                }
            }
        }

        return discoveredGraph;
    }
}
