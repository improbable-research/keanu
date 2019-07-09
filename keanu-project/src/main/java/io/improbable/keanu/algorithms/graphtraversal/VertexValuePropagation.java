package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.Probabilistic;
import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * This class enables efficient propagation of vertex updates.
 * Cascade is forward propagation and eval/lazyEval is backwards
 * propagation of updates.
 */
public class VertexValuePropagation {

    private VertexValuePropagation() {
    }

    public static void cascadeUpdate(Vertex... cascadeFrom) {
        cascadeUpdate(Arrays.asList(cascadeFrom));
    }

    public static void cascadeUpdate(Vertex vertex) {
        cascadeUpdate(Collections.singletonList(vertex));
    }

    /**
     * @param cascadeFrom A collection that contains the vertices that have been updated.
     */
    public static void cascadeUpdate(Collection<? extends Vertex> cascadeFrom) {

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparing(Vertex::getId, Comparator.naturalOrder()));
        priorityQueue.addAll(cascadeFrom);

        HashSet<Vertex> alreadyQueued = new HashSet<>(cascadeFrom);

        while (!priorityQueue.isEmpty()) {
            Vertex<?> visiting = priorityQueue.poll();

            updateVertexValue(visiting);

            for (Vertex<?> child : visiting.getChildren()) {

                if (!child.isProbabilistic() && !alreadyQueued.contains(child)) {
                    priorityQueue.offer(child);
                    alreadyQueued.add(child);
                }
            }
        }
    }

    public static void eval(Vertex... vertices) {
        eval(Arrays.asList(vertices));
    }

    public static void eval(Collection<? extends Vertex> vertices) {
        Deque<Vertex> stack = asDeque(vertices);

        Set<Vertex<?>> hasCalculated = new HashSet<>();

        while (!stack.isEmpty()) {

            Vertex<?> head = stack.peek();
            Set<Vertex<?>> parentsThatAreNotYetCalculated = parentsThatAreNotCalculated(hasCalculated, head.getParents());

            if (head.isProbabilistic() || parentsThatAreNotYetCalculated.isEmpty()) {

                Vertex<?> top = stack.pop();
                updateVertexValue(top);
                hasCalculated.add(top);

            } else {

                for (Vertex<?> vertex : parentsThatAreNotYetCalculated) {
                    stack.push(vertex);
                }

            }

        }
    }

    private static Set<Vertex<?>> parentsThatAreNotCalculated(Set<Vertex<?>> calculated, Collection<Vertex> parents) {
        Set<Vertex<?>> notCalculatedParents = new HashSet<>();
        for (Vertex<?> next : parents) {
            if (!calculated.contains(next)) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }

    public static void lazyEval(Vertex... vertices) {
        lazyEval(Arrays.asList(vertices));
    }

    public static void lazyEval(Collection<? extends Vertex> vertices) {
        Deque<Vertex> stack = asDeque(vertices);

        while (!stack.isEmpty()) {

            Vertex<?> head = stack.peek();
            Set<Vertex<?>> parentsThatAreNotYetCalculated = parentsThatAreNotCalculated(head.getParents());

            if (head.isProbabilistic() || parentsThatAreNotYetCalculated.isEmpty()) {

                Vertex<?> top = stack.pop();
                updateVertexValue(top);

            } else {

                for (Vertex<?> vertex : parentsThatAreNotYetCalculated) {
                    stack.push(vertex);
                }

            }

        }
    }

    private static Set<Vertex<?>> parentsThatAreNotCalculated(Collection<Vertex> parents) {
        Set<Vertex<?>> notCalculatedParents = new HashSet<>();
        for (Vertex<?> next : parents) {
            if (!next.hasValue()) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }

    private static Deque<Vertex> asDeque(Iterable<? extends Vertex> vertices) {
        Deque<Vertex> stack = new ArrayDeque<>();
        for (Vertex<?> v : vertices) {
            stack.push(v);
        }
        return stack;
    }

    private static <T> void updateVertexValue(Vertex<T> vertex) {
        if (vertex.isProbabilistic()) {
            if (!vertex.hasValue()) {
                vertex.setValue(((Probabilistic<T>) vertex).sample());
            }
        } else {
            if (!vertex.isObserved()) {
                vertex.setValue(((NonProbabilistic<T>) vertex).calculate());
            }
        }
    }
}
