package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.IVertex;
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

    public static void cascadeUpdate(IVertex... cascadeFrom) {
        cascadeUpdate(Arrays.asList(cascadeFrom));
    }

    public static void cascadeUpdate(IVertex vertex) {
        cascadeUpdate(Collections.singletonList(vertex));
    }

    /**
     * @param cascadeFrom A collection that contains the vertices that have been updated.
     */
    public static void cascadeUpdate(Collection<? extends IVertex> cascadeFrom) {

        PriorityQueue<IVertex> priorityQueue = new PriorityQueue<>(Comparator.comparing(IVertex::getId, Comparator.naturalOrder()));
        priorityQueue.addAll(cascadeFrom);

        HashSet<IVertex> alreadyQueued = new HashSet<>(cascadeFrom);

        while (!priorityQueue.isEmpty()) {
            IVertex<?> visiting = priorityQueue.poll();

            updateVertexValue(visiting);

            for (IVertex<?> child : visiting.getChildren()) {

                if (!child.isProbabilistic() && !alreadyQueued.contains(child)) {
                    priorityQueue.offer(child);
                    alreadyQueued.add(child);
                }
            }
        }
    }

    public static void eval(IVertex... vertices) {
        eval(Arrays.asList(vertices));
    }

    public static void eval(Collection<? extends IVertex> vertices) {
        Deque<IVertex> stack = asDeque(vertices);

        Set<IVertex<?>> hasCalculated = new HashSet<>();

        while (!stack.isEmpty()) {

            IVertex<?> head = stack.peek();
            Set<IVertex<?>> parentsThatAreNotYetCalculated = parentsThatAreNotCalculated(hasCalculated, head.getParents());

            if (head.isProbabilistic() || parentsThatAreNotYetCalculated.isEmpty()) {

                IVertex<?> top = stack.pop();
                updateVertexValue(top);
                hasCalculated.add(top);

            } else {

                for (IVertex<?> vertex : parentsThatAreNotYetCalculated) {
                    stack.push(vertex);
                }

            }

        }
    }

    private static Set<IVertex<?>> parentsThatAreNotCalculated(Set<IVertex<?>> calculated, Collection<IVertex> parents) {
        Set<IVertex<?>> notCalculatedParents = new HashSet<>();
        for (IVertex<?> next : parents) {
            if (!calculated.contains(next)) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }

    public static void lazyEval(IVertex... vertices) {
        lazyEval(Arrays.asList(vertices));
    }

    public static void lazyEval(Collection<? extends IVertex> vertices) {
        Deque<IVertex> stack = asDeque(vertices);

        while (!stack.isEmpty()) {

            IVertex<?> head = stack.peek();
            Set<IVertex<?>> parentsThatAreNotYetCalculated = parentsThatAreNotCalculated(head.getParents());

            if (head.isProbabilistic() || parentsThatAreNotYetCalculated.isEmpty()) {

                IVertex<?> top = stack.pop();
                updateVertexValue(top);

            } else {

                for (IVertex<?> vertex : parentsThatAreNotYetCalculated) {
                    stack.push(vertex);
                }

            }

        }
    }

    private static Set<IVertex<?>> parentsThatAreNotCalculated(Collection<IVertex> parents) {
        Set<IVertex<?>> notCalculatedParents = new HashSet<>();
        for (IVertex<?> next : parents) {
            if (!next.hasValue()) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }

    private static Deque<IVertex> asDeque(Iterable<? extends IVertex> vertices) {
        Deque<IVertex> stack = new ArrayDeque<>();
        for (IVertex<?> v : vertices) {
            stack.push(v);
        }
        return stack;
    }

    private static <T> void updateVertexValue(IVertex<T> vertex) {
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
