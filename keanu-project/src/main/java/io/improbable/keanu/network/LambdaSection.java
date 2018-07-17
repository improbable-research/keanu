package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

@Value
public class LambdaSection {

    private static final Predicate<Vertex> ADD_ALL = vertex -> true;
    private static final Predicate<Vertex> PROBABILISTIC_ONLY = Vertex::isProbabilistic;

    private final Set<Vertex> allVertices;
    private final Set<Vertex> probabilisticVertices;

    private LambdaSection(Set<Vertex> allVertices) {
        this.allVertices = allVertices;
        this.probabilisticVertices = allVertices.stream()
            .filter(Vertex::isProbabilistic)
            .collect(Collectors.toSet());
    }

    public static LambdaSection getUpstreamLambdaSection(Vertex<?> aVertex, boolean includeNonProbabilistic) {

        Predicate<Vertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_ONLY;

        Set<Vertex> upstreamVertices = getVerticesDepthFirst(
            aVertex,
            Vertex::getParents,
            shouldAdd
        );

        return new LambdaSection(upstreamVertices);
    }

    public static LambdaSection getDownstreamLambdaSection(Vertex<?> aVertex, boolean includeNonProbabilistic) {

        Predicate<Vertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_ONLY;

        Set<Vertex> downstreamVertices = getVerticesDepthFirst(
            aVertex,
            Vertex::getChildren,
            shouldAdd
        );

        return new LambdaSection(downstreamVertices);
    }

    /**
     * @param vertex       Vertex to start propagation from
     * @param nextVertices The next vertices to move to give a current vertex
     * @param shouldAdd    true when a give vertex should be included in the result false otherwise
     * @return A Set of vertices that are in the direction implied by nextVertices and filtered by shouldAdd
     */
    private static Set<Vertex> getVerticesDepthFirst(Vertex vertex,
                                                     Function<Vertex, Set<Vertex>> nextVertices,
                                                     Predicate<Vertex> shouldAdd) {

        Set<Vertex> visited = new HashSet<>();
        Deque<Vertex> stack = new ArrayDeque<>(nextVertices.apply(vertex));
        Set<Vertex> result = new HashSet<>();
        result.add(vertex);

        while (!stack.isEmpty()) {
            Vertex<?> visiting = stack.pop();
            visited.add(visiting);

            if (shouldAdd.test(visiting)) {
                result.add(visiting);
            }

            if (visiting.isProbabilistic()) {
                continue;
            }

            for (Vertex next : nextVertices.apply(visiting)) {
                if (!visited.contains(next)) {
                    stack.add(next);
                }
            }
        }

        return result;
    }
}
