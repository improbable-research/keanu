package io.improbable.keanu.algorithms.graphtraversal;

import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import io.improbable.keanu.vertices.Vertex;

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
    public static List<Vertex> sort(Collection<? extends Vertex> vertices) {
        return vertices.stream().
            sorted(Comparator.comparingLong(Vertex::getId))
            .collect(Collectors.toList());
    }

    public static Map<Vertex, Set<Vertex>> mapDependencies(Collection<? extends Vertex> vertices) {

        Map<Vertex, Set<Vertex>> deps = new HashMap<>();
        Set<Vertex> verticesBeingSorted = new HashSet<>(vertices);

        for (Vertex<?> v : vertices) {
            if (!deps.containsKey(v)) {
                insertParentDependencies(v, deps, verticesBeingSorted);
            }
        }

        return deps;
    }

    private static void insertParentDependencies(Vertex<?> aVertex, Map<Vertex, Set<Vertex>> dependencies, Set<Vertex> verticesToCount) {

        dependencies.computeIfAbsent(aVertex, v -> new HashSet<>());

        aVertex.getParents().forEach(p -> {
            Vertex parent = (Vertex) p;

            if (!dependencies.containsKey(parent)) {
                insertParentDependencies(parent, dependencies, verticesToCount);
            }

            final Set<Vertex> parentDependencies = dependencies.get(parent);

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
