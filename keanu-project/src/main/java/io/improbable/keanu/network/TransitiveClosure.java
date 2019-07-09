package io.improbable.keanu.network;

import io.improbable.keanu.vertices.IVertex;
import lombok.Value;

import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;

import static io.improbable.keanu.network.Propagation.getVertices;

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

    private static final Predicate<IVertex> ADD_ALL = vertex -> true;
    private static final Predicate<IVertex> PROBABILISTIC_OR_OBSERVED_ONLY = vertex -> vertex.isObserved() || vertex.isProbabilistic();

    private final Set<IVertex> allVertices;
    private final Set<IVertex> latentAndObservedVertices;

    private TransitiveClosure(Set<IVertex> allVertices) {
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
    public static TransitiveClosure getUpstreamVertices(IVertex<?> aVertex, boolean includeNonProbabilistic) {
        return getUpstreamVerticesForCollection(Collections.singletonList(aVertex), includeNonProbabilistic);
    }

    /**
     * @param aVertex                 the starting vertex
     * @param includeNonProbabilistic false if only the probabilistic and observed are wanted
     * @return All downstream vertices, not including non probabilistic if includeNonProbabilistic is false.
     */
    public static TransitiveClosure getDownstreamVertices(IVertex<?> aVertex, boolean includeNonProbabilistic) {
        return getDownstreamVerticesForCollection(Collections.singletonList(aVertex), includeNonProbabilistic);
    }

    /**
     * @param vertices                the starting vertices
     * @param includeNonProbabilistic false if only the probabilistic or observed vertices are wanted
     * @return All upstream vertices, not including non probabilistic if includeNonProbabilistic is false.
     */
    public static TransitiveClosure getUpstreamVerticesForCollection(List<IVertex> vertices, boolean includeNonProbabilistic) {

        Predicate<IVertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_OR_OBSERVED_ONLY;

        Set<IVertex> upstreamVertices = getVertices(
            vertices,
            IVertex::getParents,
            v -> false,
            shouldAdd
        );

        return new TransitiveClosure(upstreamVertices);
    }

    /**
     * @param vertices                the starting vertices
     * @param includeNonProbabilistic false if only the probabilistic or observed vertices are wanted
     * @return All upstream vertices, not including non probabilistic if includeNonProbabilistic is false.
     */
    public static TransitiveClosure getDownstreamVerticesForCollection(List<IVertex> vertices, boolean includeNonProbabilistic) {

        Predicate<IVertex> shouldAdd = includeNonProbabilistic ? ADD_ALL : PROBABILISTIC_OR_OBSERVED_ONLY;

        Set<IVertex> downstreamVertices = getVertices(
            vertices,
            IVertex::getChildren,
            v -> false,
            shouldAdd
        );

        return new TransitiveClosure(downstreamVertices);
    }

}
