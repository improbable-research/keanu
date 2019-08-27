package io.improbable.keanu.network;

import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;

@UtilityClass
public class Propagation {

    public static Set<Vertex> getVertices(Vertex vertex, Function<Vertex, Collection<Vertex>> nextVertices, Function<Vertex, Boolean> stoppingCondition,
                                          Predicate<Vertex> shouldAdd) {
        return getVertices(Collections.singletonList(vertex), nextVertices, stoppingCondition, shouldAdd);
    }

    /**
     * @param vertices          vertices to start propagation from
     * @param nextVertices      The next vertices to move to given a current vertex. E.g getChildren for downstream or
     *                          getParents for upstream.
     * @param shouldAdd         true when a given vertex should be included in the result, false otherwise
     * @param stoppingCondition true when a given vertex should be stopped at
     * @return A Set of vertices that are in the direction implied by nextVertices and filtered by shouldAdd
     */
    public static Set<Vertex> getVertices(List<Vertex> vertices,
                                          Function<Vertex, Collection<Vertex>> nextVertices,
                                          Function<Vertex, Boolean> stoppingCondition,
                                          Predicate<Vertex> shouldAdd) {

        Set<Vertex> nextAll = vertices.stream()
            .flatMap(v -> nextVertices.apply(v).stream())
            .collect(Collectors.toSet());

        Deque<Vertex> stack = new ArrayDeque<>(nextAll);
        Set<Vertex> queued = new HashSet<>(vertices);
        queued.addAll(nextAll);

        Set<Vertex> result = new HashSet<>(vertices);

        while (!stack.isEmpty()) {
            Vertex visiting = stack.pop();

            if (shouldAdd.test(visiting)) {
                result.add(visiting);
            }

            if (stoppingCondition.apply(visiting)) {
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
