package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;

import java.util.*;
import java.util.stream.Collectors;

public class TopologicalSort {

    private TopologicalSort() {
    }

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

        aVertex.getParents().forEach(parent -> {

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
