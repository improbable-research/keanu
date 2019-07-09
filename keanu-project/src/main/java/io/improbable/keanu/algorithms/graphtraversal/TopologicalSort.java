package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.IVertex;

import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class TopologicalSort {

    private TopologicalSort() {
    }

    /**
     * This algorithm returns a linear ordering of vertices such that for every edge uv from
     * vertex u to vertex v; u comes before v in the ordering.
     *
     * @param vertices the vertices to sort
     * @return a linear ordering of vertices by order of execution
     */
    public static List<IVertex> sort(Collection<? extends IVertex> vertices) {
        return vertices
            .stream()
            .sorted(Comparator.comparing(IVertex::getId, Comparator.naturalOrder()))
            .collect(Collectors.toList());
    }

    public static Map<IVertex, Set<IVertex>> mapDependencies(Collection<? extends IVertex> vertices) {

        Map<IVertex, Set<IVertex>> deps = new HashMap<>();
        Set<IVertex> verticesBeingSorted = new HashSet<>(vertices);

        for (IVertex<?> v : vertices) {
            if (!deps.containsKey(v)) {
                insertParentDependencies(v, deps, verticesBeingSorted);
            }
        }

        return deps;
    }

    private static void insertParentDependencies(IVertex<?> aVertex, Map<IVertex, Set<IVertex>> dependencies, Set<IVertex> verticesToCount) {

        dependencies.computeIfAbsent(aVertex, v -> new HashSet<>());

        aVertex.getParents().forEach(parent -> {

            if (!dependencies.containsKey(parent)) {
                insertParentDependencies(parent, dependencies, verticesToCount);
            }

            final Set<IVertex> parentDependencies = dependencies.get(parent);

            dependencies.computeIfPresent(aVertex, (vertex, vertexDependencies) -> {
                vertexDependencies.addAll(parentDependencies);
                if (verticesToCount.contains(parent)) {
                    vertexDependencies.add(parent);
                }
                return vertexDependencies;
            });
        });

    }
}
