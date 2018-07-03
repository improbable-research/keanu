package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

public class MarkovBlanket {

    private MarkovBlanket() {
    }

    /**
     * This algorithm finds all of the vertices that shield it from the rest of the Bayesian Network.
     * By knowing the Markov Blanket of a vertex, we can fully predict the behaviour of that vertex.
     *
     *
     *
     * @param aVertex the vertex to find the Markov Blanket for
     * @return A set of vertices that are affected by, or affect, a given vertex
     */
    public static Set<Vertex> get(Vertex<?> aVertex) {

        Set<Vertex> parents = getUpstreamProbabilisticVertices(aVertex);
        Set<Vertex> children = getDownstreamProbabilisticVertices(aVertex);
        Set<Vertex> childrensParents = getUpstreamProbabilisticVertices(children);

        Set<Vertex> blanket = new HashSet<>();
        blanket.addAll(parents);
        blanket.addAll(children);
        blanket.addAll(childrensParents);

        blanket.remove(aVertex);

        return blanket;
    }

    private static Set<Vertex> getUpstreamProbabilisticVertices(Vertex<?> aVertex) {
        return getUpstreamProbabilisticVertices(aVertex, new HashSet<>(), new HashSet<>());
    }

    private static Set<Vertex> getUpstreamProbabilisticVertices(Vertex<?> aVertex, Set<Vertex> probabilistic, Set<Vertex> visited) {
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

    private static Set<Vertex> getUpstreamProbabilisticVertices(Collection<Vertex> vertices) {

        Set<Vertex> visited = new HashSet<>();
        Set<Vertex> probabilistic = new HashSet<>();

        for (Vertex<?> vertex : vertices) {
            getUpstreamProbabilisticVertices(vertex, probabilistic, visited);
        }

        return probabilistic;
    }

    public static Set<Vertex> getDownstreamProbabilisticVertices(Vertex<?> aVertex) {
        return getDownstreamProbabilisticVertices(aVertex, new HashSet<>(), new HashSet<>());
    }

    private static Set<Vertex> getDownstreamProbabilisticVertices(Vertex<?> aVertex, Set<Vertex> probabilistic, Set<Vertex> visited) {
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
