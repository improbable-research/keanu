package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import lombok.Value;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import static io.improbable.keanu.network.Propagation.getVertices;

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
        return getUpstreamLambdaSectionForCollection(Collections.singletonList(aVertex), includeNonProbabilistic);
    }

    /**
     * @param aVertex                 the starting vertex
     * @param includeNonProbabilistic false if only the probabilistic and observed are wanted
     * @return All downstream vertices up to probabilistic or observed vertices if includeNonProbabilistic
     * is true. All downstream probabilistic or observed vertices stopping at probabilistic or observed if
     * includeNonProbabilistic is false.
     */
    public static LambdaSection getDownstreamLambdaSection(Vertex<?> aVertex, boolean includeNonProbabilistic) {
        return getDownstreamLambdaSectionForCollection(Collections.singletonList(aVertex), includeNonProbabilistic);
    }

    /**
     * @param vertices                the starting vertices
     * @param includeNonProbabilistic false if only the probabilistic or observed vertices are wanted
     * @return All upstream vertices up to probabilistic or observed vertices if includeNonProbabilistic
     * is true. All upstream probabilistic or observed vertices stopping at probabilistic or observed if
     * includeNonProbabilistic is false.
     */
    public static LambdaSection getUpstreamLambdaSectionForCollection(List<Vertex> vertices, boolean includeNonProbabilistic) {

        Predicate<Vertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_OR_OBSERVED_ONLY;

        Set<Vertex> upstreamVertices = getVertices(
            vertices,
            Vertex::getParents,
            v -> v.isObserved() || v.isProbabilistic(),
            shouldAdd
        );

        return new LambdaSection(upstreamVertices);
    }

    /**
     * @param vertices                the starting vertices
     * @param includeNonProbabilistic false if only the probabilistic or observed vertices are wanted
     * @return All upstream vertices up to probabilistic or observed vertices if includeNonProbabilistic
     * is true. All upstream probabilistic or observed vertices stopping at probabilistic or observed if
     * includeNonProbabilistic is false.
     */
    public static LambdaSection getDownstreamLambdaSectionForCollection(List<Vertex> vertices, boolean includeNonProbabilistic) {

        Predicate<Vertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_OR_OBSERVED_ONLY;

        Set<Vertex> downstreamVertices = getVertices(
            vertices,
            Vertex::getChildren,
            v -> v.isObserved() || v.isProbabilistic(),
            shouldAdd
        );

        return new LambdaSection(downstreamVertices);
    }
}
