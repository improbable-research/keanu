package io.improbable.keanu.util.graph;

import java.awt.*;
import java.util.*;
import java.util.function.Predicate;

/**
 * This class represents a Graph (collection of nodes and edges)
 * @param <N> The nodes of the graph
 * @param <E> The edges of the graph
 */
public abstract class AbstractGraph<N extends GraphNode, E extends GraphEdge<N>> {

    protected Map<String, String> metadata = new HashMap<>();
    protected Set<E> edges = new HashSet<>();

    /**
     * @return the metadata collection for the graph
     */
    public Map<String, String> getMetaData() {
        return metadata;
    }

    /**
     * @return All the nodes in the graph
     */
    public abstract Collection<N> getNodes();

    /**
     * Method called before any writing is done - this can be used for lazy formatting
     */
    public void prepareForExport(){

    }

    /**
     * @return All the edges in the graph
     */
    public Collection<E> getEdges() {
        return edges;
    }

    /**
     * This converts a java color to a .dot format
     * @param c The color to convert
     * @return the colot as a 6 digit hex string (e.g. #FF00FF)
     */
    public static final String formatColorForDot(Color c) {
        if (c == null) return null;
        return String.format("#%06X", (0xFFFFFF & c.getRGB()));
    }

    public Set<E> findEdges(Predicate<E> f) {
        Set<E> edges = new HashSet<>();
        for (E e : getEdges()) {
            if (f.test(e)) {
                edges.add(e);
            }
        }
        return edges;

    }

    public Set<E> findEdgesUsing(N n) {
        return findEdges((v) -> v.getSource() == n || v.getDestination() == n);
    }

    public Set<E> findEdgesFrom(N n) {
        return findEdges((v) -> v.getSource() == n);
    }

    public Set<E> findEdgesTo(N n) {
        return findEdges((v) -> v.getDestination() == n);
    }

    public int edgeCount() {
        return getEdges().size();
    }

    public int nodeCount() {
        return getNodes().size();
    }

}
