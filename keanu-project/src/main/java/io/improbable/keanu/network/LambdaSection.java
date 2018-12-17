package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

/**
 * A Lambda Section is defined as a given vertex and all the vertices that it affects (downstream) OR
 * all of the vertices that affects it (upstream), stopping at observed or probabilistic vertices.
 * <p>
 * For example:
 * <p>
 * A = SomeDistribution(...)
 * B = A.cos()
 * C = SomeDistribution(B, ...)
 * D = C.times(2)
 * <p>
 * The downstream Lambda Section of A would be [A, B, C]
 * The upstream Lambda Section of D would be [D, C]
 * The upstream Lambda Section of C would be [C, B, A]
 */
@Value
public class LambdaSection {

    private static final Predicate<Vertex> ADD_ALL = vertex -> true;
    private static final Predicate<Vertex> PROBABILISTIC_OR_OBSERVED_ONLY = vertex -> vertex.isObserved() || vertex.isProbabilistic();

    private final Set<Vertex> allVertices;
    private final Set<Vertex> latentAndObservedVertices;

    private LambdaSection(Set<Vertex> allVertices) {
        this.allVertices = allVertices;
        this.latentAndObservedVertices = allVertices.stream()
            .filter(PROBABILISTIC_OR_OBSERVED_ONLY)
            .collect(Collectors.toSet());
    }

    /**
     * @param aVertex                 the starting vertex
     * @param includeNonProbabilistic false if only the probabilistic or observed vertices are wanted
     * @return All upstream vertices up to probabilistic or observed vertices if includeNonProbabilistic
     * is true. All upstream probabilistic or observed vertices stopping at probabilistic or observed if
     * includeNonProbabilistic is false.
     */
    public static LambdaSection getUpstreamLambdaSection(Vertex<?> aVertex, boolean includeNonProbabilistic) {

        Predicate<Vertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_OR_OBSERVED_ONLY;

        Set<Vertex> upstreamVertices = getVerticesDepthFirst(
            aVertex,
            Vertex::getParents,
            shouldAdd
        );

        return new LambdaSection(upstreamVertices);
    }

    /**
     * @param aVertex                 the starting vertex
     * @param includeNonProbabilistic false if only the probabilistic and observed are wanted
     * @return All downstream vertices up to probabilistic or observed vertices if includeNonProbabilistic
     * is true. All downstream probabilistic or observed vertices stopping at probabilistic or observed if
     * includeNonProbabilistic is false.
     */
    public static LambdaSection getDownstreamLambdaSection(Vertex<?> aVertex, boolean includeNonProbabilistic) {

        Predicate<Vertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_OR_OBSERVED_ONLY;

        Set<Vertex> downstreamVertices = getVerticesDepthFirst(
            aVertex,
            Vertex::getChildren,
            shouldAdd
        );

        return new LambdaSection(downstreamVertices);
    }

    /**
     * @param vertex       Vertex to start propagation from
     * @param nextVertices The next vertices to move to given a current vertex. E.g getChildren for downstream or
     *                     getParents for upstream.
     * @param shouldAdd    true when a give vertex should be included in the result false otherwise
     * @return A Set of vertices that are in the direction implied by nextVertices and filtered by shouldAdd
     */
    public static Set<Vertex> getVerticesDepthFirst(Vertex vertex,
                                                    Function<Vertex, Collection<Vertex>> nextVertices,
                                                    Predicate<Vertex> shouldAdd) {

        Collection<Vertex> initialNext = nextVertices.apply(vertex);
        Set<Vertex> queued = new HashSet<>(initialNext);
        Deque<Vertex> stack = new ArrayDeque<>(initialNext);
        Set<Vertex> result = new HashSet<>();
        result.add(vertex);

        while (!stack.isEmpty()) {
            Vertex visiting = stack.pop();

            if (shouldAdd.test(visiting)) {
                result.add(visiting);
            }

            if (visiting.isObserved() || visiting.isProbabilistic()) {
                continue;
            }

            for (Vertex next : nextVertices.apply(visiting)) {
                if (!queued.contains(next)) {
                    stack.add(next);
                    queued.add(next);
                }
            }
        }

        return result;
    }
}
