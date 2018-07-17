package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.network.LambdaSection;
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
     * @param aVertex the vertex to find the Markov Blanket for
     * @return A set of vertices that are affected by, or affect, a given vertex
     */
    public static Set<Vertex> get(Vertex<?> aVertex) {

        LambdaSection parents = LambdaSection.getUpstreamLambdaSection(aVertex, false);
        LambdaSection children = LambdaSection.getDownstreamLambdaSection(aVertex, false);
        Set<Vertex> childrensParents = getUpstreamProbabilisticVertices(children.getProbabilisticVertices());

        Set<Vertex> blanket = new HashSet<>();
        blanket.addAll(parents.getProbabilisticVertices());
        blanket.addAll(children.getProbabilisticVertices());
        blanket.addAll(childrensParents);

        blanket.remove(aVertex);

        return blanket;
    }

    private static Set<Vertex> getUpstreamProbabilisticVertices(Collection<Vertex> vertices) {

        Set<Vertex> probabilistic = new HashSet<>();

        for (Vertex<?> vertex : vertices) {
            LambdaSection upstreamLambdaSection = LambdaSection.getUpstreamLambdaSection(vertex, false);
            probabilistic.addAll(upstreamLambdaSection.getProbabilisticVertices());
        }

        return probabilistic;
    }

}
