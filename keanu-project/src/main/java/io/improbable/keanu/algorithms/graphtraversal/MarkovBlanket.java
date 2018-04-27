package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class MarkovBlanket {

    private MarkovBlanket() {
    }

    public static Set<Vertex<?>> get(Vertex<?> aVertex) {

        Set<Vertex<?>> parents = getUpstreamProbabilisticVertices(aVertex);
        Set<Vertex<?>> children = getDownstreamProbabilisticVertices(aVertex);
        Set<Vertex<?>> childrensParents = getUpstreamProbabilisticVertices(children);

        Set<Vertex<?>> blanket = new HashSet<>();
        blanket.addAll(parents);
        blanket.addAll(children);
        blanket.addAll(childrensParents);

        blanket.remove(aVertex);

        return blanket;
    }

    public static Set<Vertex<?>> getUpstreamProbabilisticVertices(Vertex<?> aVertex) {
        return getUpstreamProbabilisticVertices(aVertex, new HashSet<>(), new HashSet<>());
    }

    private static Set<Vertex<?>> getUpstreamProbabilisticVertices(Vertex<?> aVertex, Set<Vertex<?>> probabilistic, Set<Vertex<?>> visited) {
        visited.add(aVertex);

        aVertex.getParents().forEach(parent -> {
            if (!visited.contains(parent)) {
                if (parent.isProbabilistic()) {
                    probabilistic.add(parent);
                } else {
                    getUpstreamProbabilisticVertices(parent, probabilistic, visited);
                }
            }
        });

        return probabilistic;
    }

    public static Set<Vertex<?>> getUpstreamProbabilisticVertices(Collection<Vertex<?>> vertices) {

        Set<Vertex<?>> visited = new HashSet<>();
        Set<Vertex<?>> probabilistic = new HashSet<>();

        for (Vertex<?> vertex : vertices) {
            getUpstreamProbabilisticVertices(vertex, probabilistic, visited);
        }

        return probabilistic;
    }

    public static Set<Vertex<?>> getDownstreamProbabilisticVertices(Vertex<?> aVertex) {
        return getDownstreamProbabilisticVertices(aVertex, new HashSet<>(), new HashSet<>());
    }

    private static Set<Vertex<?>> getDownstreamProbabilisticVertices(Vertex<?> aVertex, Set<Vertex<?>> probabilistic, Set<Vertex<?>> visited) {
        visited.add(aVertex);

        aVertex.getChildren().forEach(child -> {
            if (!visited.contains(child)) {
                if (child.isProbabilistic()) {
                    probabilistic.add(child);
                } else {
                    getDownstreamProbabilisticVertices(child, probabilistic, visited);
                }
            }
        });

        return probabilistic;
    }

}
