package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.vertices.Vertex;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

public class LatentIncrementSort {

    private LatentIncrementSort() {}

    /**
     * Orders observed vertices by the smallest increment of additional latent vertices upstream of
     * the observed vertex.
     *
     * @param vertices vertices to sort
     * @return Sorted observed vertices
     */
    public static Map<Vertex, Set<Vertex>> sort(Collection<? extends Vertex> vertices) {

        Map<Vertex, Set<Vertex>> dependencies = getObservedVertexLatentDependencies(vertices);
        Map<Vertex, Set<Vertex>> dependants = mapDependents(dependencies);
        LinkedHashMap<Vertex, Set<Vertex>> observedVertexOrder = new LinkedHashMap<>();
        List<Vertex<?>> verticesWithFewestDependencies;

        while (!(verticesWithFewestDependencies = getVerticesWithFewestDependencies(dependencies))
                .isEmpty()) {
            Vertex vertex = verticesWithFewestDependencies.get(0);
            Set<Vertex> vertexDependencies = dependencies.remove(vertex);
            observedVertexOrder.put(vertex, vertexDependencies);

            for (Vertex<?> upstreamVertex : vertexDependencies) {
                removeDependencyFromOtherVertices(upstreamVertex, dependants, dependencies);
            }
        }

        return observedVertexOrder;
    }

    private static Map<Vertex, Set<Vertex>> getObservedVertexLatentDependencies(
            Collection<? extends Vertex> vertices) {

        Map<Vertex, Set<Vertex>> dependencies = TopologicalSort.mapDependencies(vertices);
        Map<Vertex, Set<Vertex>> observedVertexLatentDependencies = new HashMap<>();

        for (Map.Entry<Vertex, Set<Vertex>> entry : dependencies.entrySet()) {
            Vertex<?> vertex = entry.getKey();

            if (vertex.isObserved()) {
                Set<Vertex> vertexDependencies = entry.getValue();
                Set<Vertex> latentDependencies = getLatentDependencies(vertexDependencies);
                observedVertexLatentDependencies.put(vertex, latentDependencies);
            }
        }

        return observedVertexLatentDependencies;
    }

    private static Set<Vertex> getLatentDependencies(Set<Vertex> dependencies) {
        return dependencies
                .stream()
                .filter(v -> v.isProbabilistic() && !v.isObserved())
                .collect(Collectors.toSet());
    }

    private static Map<Vertex, Set<Vertex>> mapDependents(Map<Vertex, Set<Vertex>> dependencies) {

        Map<Vertex, Set<Vertex>> dependants = new HashMap<>();
        for (Map.Entry<Vertex, Set<Vertex>> entry : dependencies.entrySet()) {
            Vertex<?> dependant = entry.getKey();
            for (Vertex<?> vertex : entry.getValue()) {
                dependants.computeIfAbsent(vertex, v -> dependants.put(v, new HashSet<>()));
                dependants.get(vertex).add(dependant);
            }
        }

        return dependants;
    }

    private static List<Vertex<?>> getVerticesWithFewestDependencies(
            Map<Vertex, Set<Vertex>> dependencies) {

        List<Vertex<?>> verticesWithFewestDependencies = new ArrayList<>();
        int minDependencies = Integer.MAX_VALUE;

        for (Map.Entry<Vertex, Set<Vertex>> entry : dependencies.entrySet()) {
            Vertex<?> v = entry.getKey();
            int dependsOn = entry.getValue().size();
            if (dependsOn < minDependencies) {
                minDependencies = dependsOn;
                verticesWithFewestDependencies.clear();
                verticesWithFewestDependencies.add(v);
            } else if (dependsOn == minDependencies) {
                verticesWithFewestDependencies.add(v);
            }
        }

        return verticesWithFewestDependencies;
    }

    private static void removeDependencyFromOtherVertices(
            Vertex<?> vertex,
            Map<Vertex, Set<Vertex>> dependants,
            Map<Vertex, Set<Vertex>> dependencies) {

        dependants
                .get(vertex)
                .forEach(
                        dependant -> {
                            if (dependencies.containsKey(dependant)) {
                                dependencies.get(dependant).remove(vertex);
                            }
                        });
    }
}
