package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import static io.improbable.keanu.network.LambdaSection.getVertices;

/**
 * A Transitive Closure is defined as a given vertex and all the vertices that it affects (downstream) OR
 * all of the vertices that affects it (upstream). Unlike a Lambda Section it does not stop at observed or probabilistic vertices.
 * <p>
 * For example:
 * <p>
 * A = SomeDistribution(...)
 * B = A.cos()
 * C = SomeDistribution(B, ...)
 * D = C.times(2)
 * <p>
 * The downstream Transitive Closure of A would be [A, B, C, D]
 * The upstream Transitive Closure of D would be [D, C, B, A]
 * The upstream Transitive Closure of C would be [C, B, A]
 */
@Value
public class TransitiveClosure {

    private static final Predicate<Vertex> ADD_ALL = vertex -> true;
    private static final Predicate<Vertex> PROBABILISTIC_OR_OBSERVED_ONLY = vertex -> vertex.isObserved() || vertex.isProbabilistic();

    private final Set<Vertex> allVertices;
    private final Set<Vertex> latentAndObservedVertices;

    private TransitiveClosure(Set<Vertex> allVertices) {
        this.allVertices = allVertices;
        this.latentAndObservedVertices = allVertices.stream()
            .filter(PROBABILISTIC_OR_OBSERVED_ONLY)
            .collect(Collectors.toSet());
    }

    /**
     * @param aVertex                 the starting vertex
     * @param includeNonProbabilistic false if only the probabilistic or observed vertices are wanted
     * @return All upstream vertices, not including non probabilistic if includeNonProbabilistic is false.
     */
    public static TransitiveClosure getUpstreamVertices(Vertex<?> aVertex, boolean includeNonProbabilistic) {
        return getUpstreamVerticesForCollection(Collections.singletonList(aVertex), includeNonProbabilistic);
    }

    /**
     * @param aVertex                 the starting vertex
     * @param includeNonProbabilistic false if only the probabilistic and observed are wanted
     * @return All downstream vertices, not including non probabilistic if includeNonProbabilistic is false.
     */
    public static TransitiveClosure getDownstreamVertices(Vertex<?> aVertex, boolean includeNonProbabilistic) {
        return getDownstreamVerticesForCollection(Collections.singletonList(aVertex), includeNonProbabilistic);
    }

    /**
     * @param vertices                the starting vertices
     * @param includeNonProbabilistic false if only the probabilistic or observed vertices are wanted
     * @return  All upstream vertices, not including non probabilistic if includeNonProbabilistic is false.
     */
    public static TransitiveClosure getUpstreamVerticesForCollection(List<Vertex> vertices, boolean includeNonProbabilistic) {

        Predicate<Vertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_OR_OBSERVED_ONLY;

        Set<Vertex> upstreamVertices = getVertices(
            vertices,
            Vertex::getParents,
            v -> false,
            shouldAdd
        );

        return new TransitiveClosure(upstreamVertices);
    }

    /**
     * @param vertices                the starting vertices
     * @param includeNonProbabilistic false if only the probabilistic or observed vertices are wanted
     * @return  All upstream vertices, not including non probabilistic if includeNonProbabilistic is false.
     */
    public static TransitiveClosure getDownstreamVerticesForCollection(List<Vertex> vertices, boolean includeNonProbabilistic) {

        Predicate<Vertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_OR_OBSERVED_ONLY;

        Set<Vertex> downstreamVertices = getVertices(
            vertices,
            Vertex::getChildren,
            v -> false,
            shouldAdd
        );

        return new TransitiveClosure(downstreamVertices);
    }

}
