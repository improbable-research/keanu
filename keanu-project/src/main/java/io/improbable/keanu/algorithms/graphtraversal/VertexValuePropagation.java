package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;

import java.util.*;

/**
 * This class enables efficient propagation of vertex updates. For
 * forward propagation it makes use of a pre-evaluation graph
 * traversal that allows the evaluation to not redo work and execute
 * in the correct order. This is done without mutating the graph.
 */
public class VertexValuePropagation {

    private VertexValuePropagation() {
    }

    public static Map<String, Long> exploreSetting(Vertex<?> toBeSet) {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        stack.push(toBeSet);
        return exploreSetting(stack);
    }

    public static Map<String, Long> exploreSetting(Collection<? extends Vertex<?>> toBeSet) {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        for (Vertex<?> v : toBeSet) {
            stack.push(v);
        }
        return exploreSetting(stack);
    }

    /**
     * This explores the graph and finds the number of times a vertex would be
     * visited upon propagation of changes to the vertices specified in the initial
     * stack.
     * <p>
     * This does a depth first traversal of the graph starting from the vertices
     * in the stack. It does not revisit vertices but does track which ones, and
     * how many times, they would be visited.
     *
     * @param stack a Stack containing the vertices to be set
     * @return a map containing the vertex id as a key and the number of times
     * to visit before vertex evaluation as the value of the map.
     */
    private static Map<String, Long> exploreSetting(Deque<Vertex<?>> stack) {

        Set<Vertex<?>> hasVisited = new HashSet<>();
        Map<String, Long> crossRoadCount = new HashMap<>();

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
                    crossRoadCount.put(child.getId(), crossRoadCount.getOrDefault(child.getId(), 0L) + 1);
                }

            }
        }

        return crossRoadCount;
    }

    public static void cascadeUpdate(Vertex<?>... updatingVertices) {
        List<Vertex<?>> vertices = Arrays.asList(updatingVertices);
        cascadeUpdate(vertices, exploreSetting(vertices));
    }

    public static void cascadeUpdate(List<? extends Vertex<?>> vertices) {
        cascadeUpdate(vertices, exploreSetting(vertices));
    }

    public static void cascadeUpdate(Vertex<?> vertex, Map<String, Long> explored) {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        stack.push(vertex);
        cascadeUpdate(stack, explored);
    }

    public static void cascadeUpdate(List<? extends Vertex<?>> vertices, Map<String, Long> explored) {
        Deque<Vertex<?>> stack = new ArrayDeque<>();
        for (Vertex<?> v : vertices) {
            stack.push(v);
        }
        cascadeUpdate(stack, explored);
    }

    /**
     * @param stack    A Stack that contains the vertices that have been updated.
     * @param explored The results of previously traversing the graph. The keys in this map are
     *                 the vertex ids and the values are how many times to visit them before
     *                 evaluating.
     */
    private static void cascadeUpdate(Deque<Vertex<?>> stack, Map<String, Long> explored) {

        Map<String, Long> turnAroundCounts = new HashMap<>(explored);

        while (!stack.isEmpty()) {
            Vertex<?> visiting = stack.pop();

            visiting.updateValue();

            for (Vertex<?> child : visiting.getChildren()) {

                if (child.isProbabilistic()) {
                    continue;
                }

                Long currentCount = turnAroundCounts.get(child.getId());

                if (currentCount != null && currentCount != 0) {
                    turnAroundCounts.put(child.getId(), currentCount - 1);
                } else {
                    stack.push(child);
                }
            }

        }
    }
}
