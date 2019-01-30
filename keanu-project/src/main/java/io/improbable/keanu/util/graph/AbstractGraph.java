package io.improbable.keanu.util.graph;

import java.util.*;
import java.util.function.Predicate;

public abstract class AbstractGraph<N extends GraphNode, E extends GraphEdge<N>> {

    protected Map<String, String> metadata = new HashMap<>();
    protected Set<E> edges = new HashSet<>();

    public Map<String, String> getMetaData() {
        return metadata;
    }

    public abstract Collection<N> getNodes();

    public void prepareForExport(){

    }

    public Collection<E> getEdges() {
        return edges;
    }

    public static void mergeMetadata(Map<String, String> target, Map<String, String>... input) {
        for (Map<String, String> i : input) {
            for (Map.Entry<String, String> e : i.entrySet()) {
                if (target.containsKey(e.getKey())) {
                    target.put(e.getKey(), target.get(e.getKey()) + ", " + e.getValue());
                } else {
                    target.put(e.getKey(), e.getValue());
                }
            }
        }
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
