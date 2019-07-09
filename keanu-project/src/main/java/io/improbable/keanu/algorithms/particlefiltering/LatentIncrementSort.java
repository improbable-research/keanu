package io.improbable.keanu.algorithms.particlefiltering;

import io.improbable.keanu.algorithms.graphtraversal.TopologicalSort;
import io.improbable.keanu.vertices.IVertex;

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

    private LatentIncrementSort() {
    }

    /**
     * Orders observed vertices by the smallest increment of additional latent vertices upstream of the observed vertex.
     *
     * @param vertices vertices to sort
     * @return Sorted observed vertices
     */
    public static Map<IVertex, Set<IVertex>> sort(Collection<? extends IVertex> vertices) {

        Map<IVertex, Set<IVertex>> dependencies = getObservedVertexLatentDependencies(vertices);
        Map<IVertex, Set<IVertex>> dependants = mapDependents(dependencies);
        LinkedHashMap<IVertex, Set<IVertex>> observedVertexOrder = new LinkedHashMap<>();
        List<IVertex<?>> verticesWithFewestDependencies;

        while (!(verticesWithFewestDependencies = getVerticesWithFewestDependencies(dependencies)).isEmpty()) {
            IVertex vertex = verticesWithFewestDependencies.get(0);
            Set<IVertex> vertexDependencies = dependencies.remove(vertex);
            observedVertexOrder.put(vertex, vertexDependencies);

            for (IVertex<?> upstreamVertex : vertexDependencies) {
                removeDependencyFromOtherVertices(upstreamVertex, dependants, dependencies);
            }
        }

        return observedVertexOrder;
    }

    private static Map<IVertex, Set<IVertex>> getObservedVertexLatentDependencies(Collection<? extends IVertex> vertices) {

        Map<IVertex, Set<IVertex>> dependencies = TopologicalSort.mapDependencies(vertices);
        Map<IVertex, Set<IVertex>> observedVertexLatentDependencies = new HashMap<>();

        for (Map.Entry<IVertex, Set<IVertex>> entry : dependencies.entrySet()) {
            IVertex<?> vertex = entry.getKey();

            if (vertex.isObserved()) {
                Set<IVertex> vertexDependencies = entry.getValue();
                Set<IVertex> latentDependencies = getLatentDependencies(vertexDependencies);
                observedVertexLatentDependencies.put(vertex, latentDependencies);
            }
        }

        return observedVertexLatentDependencies;
    }

    private static Set<IVertex> getLatentDependencies(Set<IVertex> dependencies) {
        return dependencies.stream()
            .filter(v -> v.isProbabilistic() && !v.isObserved())
            .collect(Collectors.toSet());
    }

    private static Map<IVertex, Set<IVertex>> mapDependents(Map<IVertex, Set<IVertex>> dependencies) {

        Map<IVertex, Set<IVertex>> dependants = new HashMap<>();
        for (Map.Entry<IVertex, Set<IVertex>> entry : dependencies.entrySet()) {
            IVertex<?> dependant = entry.getKey();
            for (IVertex<?> vertex : entry.getValue()) {
                dependants.computeIfAbsent(vertex, v -> dependants.put(v, new HashSet<>()));
                dependants.get(vertex).add(dependant);
            }
        }

        return dependants;
    }

    private static List<IVertex<?>> getVerticesWithFewestDependencies(Map<IVertex, Set<IVertex>> dependencies) {

        List<IVertex<?>> verticesWithFewestDependencies = new ArrayList<>();
        int minDependencies = Integer.MAX_VALUE;

        for (Map.Entry<IVertex, Set<IVertex>> entry : dependencies.entrySet()) {
            IVertex<?> v = entry.getKey();
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

    private static void removeDependencyFromOtherVertices(IVertex<?> vertex, Map<IVertex, Set<IVertex>> dependants,
                                                          Map<IVertex, Set<IVertex>> dependencies) {

        dependants.get(vertex).forEach(dependant -> {
            if (dependencies.containsKey(dependant)) {
                dependencies.get(dependant).remove(vertex);
            }
        });
    }
}
