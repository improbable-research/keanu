package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;

import java.util.*;

/**
 *
 */
public class VertexValuePropagation {

    public static void cascadeUpdate(Vertex<?>... updatingVertices) {
        List<Vertex<?>> vertices = Arrays.asList(updatingVertices);
        cascadeUpdate(vertices, exploreSetting(vertices));
    }

    public static void cascadeUpdate(List<? extends Vertex<?>> vertices) {
        cascadeUpdate(vertices, exploreSetting(vertices));
    }

    public static void cascadeUpdate(Vertex<?> vertex, Map<String, Integer> explored) {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        stack.push(vertex);
        cascadeUpdate(stack, explored);
    }

    public static void cascadeUpdate(List<? extends Vertex<?>> vertices, Map<String, Integer> explored) {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        for (Vertex<?> v : vertices) {
            stack.push(v);
        }
        cascadeUpdate(stack, explored);
    }

    private static void cascadeUpdate(Deque<Vertex<?>> stack, Map<String, Integer> explored) {

        Map<String, Integer> turnAroundCounts = new HashMap<>(explored);

        while (!stack.isEmpty()) {
            Vertex<?> visiting = stack.pop();

            visiting.updateValue();

            for (Vertex<?> child : visiting.getChildren()) {

                if (child.isProbabilistic()) {
                    continue;
                }

                Integer currentCount = turnAroundCounts.get(child.getId());

                if (currentCount != null && currentCount != 0) {
                    turnAroundCounts.put(child.getId(), currentCount - 1);
                } else {
                    stack.push(child);
                }
            }

        }
    }

    public static Map<String, Integer> exploreSetting(Vertex<?> toBeSet) {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        stack.push(toBeSet);
        return exploreSetting(stack);
    }

    public static Map<String, Integer> exploreSetting(Collection<? extends Vertex<?>> toBeSet) {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        for (Vertex<?> v : toBeSet) {
            stack.push(v);
        }
        return exploreSetting(stack);
    }

    private static Map<String, Integer> exploreSetting(Deque<Vertex<?>> stack) {

        Set<Vertex<?>> hasVisited = new HashSet<>();
        Map<String, Integer> crossRoadCount = new HashMap<>();

        while (!stack.isEmpty()) {

            Vertex<?> visiting = stack.pop();
            hasVisited.add(visiting);

            for (Vertex<?> child : visiting.getChildren()) {

                if (child.isProbabilistic()) {
                    continue;
                }

                if (!hasVisited.contains(child)) {
                    stack.push(child);
                } else {
                    crossRoadCount.put(child.getId(), crossRoadCount.getOrDefault(child.getId(), 0) + 1);
                }

            }
        }

        return crossRoadCount;
    }
}
