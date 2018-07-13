package io.improbable.keanu.algorithms.graphtraversal;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.Set;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Vertex;

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

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.comparingLong(Vertex::getId));
        priorityQueue.addAll(cascadeFrom);

        HashSet<Vertex> alreadyQueued = new HashSet<>(cascadeFrom);

        while (!priorityQueue.isEmpty()) {
            Vertex<?> visiting = priorityQueue.poll();

            visiting.updateValue();

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
        Deque<IVertex> stack = asDeque(vertices);

        Set<IVertex<?>> hasCalculated = new HashSet<>();

        while (!stack.isEmpty()) {

            IVertex<?> head = stack.peek();
            Set<IVertex<?>> parentsThatAreNotYetCalculated = parentsThatAreNotCalculated(hasCalculated, head.getParents());

            if (((Vertex)head).isProbabilistic() || parentsThatAreNotYetCalculated.isEmpty()) {

                IVertex<?> top = stack.pop();
                top.updateValue();
                hasCalculated.add(top);

            } else {

                for (IVertex<?> vertex : parentsThatAreNotYetCalculated) {
                    stack.push(vertex);
                }

            }

        }
    }

    private static Set<IVertex<?>> parentsThatAreNotCalculated(Set<IVertex<?>> calculated, Set<? extends IVertex> parents) {
        Set<IVertex<?>> notCalculatedParents = new HashSet<>();
        for (IVertex<?> next : parents) {
            if (!calculated.contains(next)) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }

    public static void lazyEval(Vertex... vertices) {
        lazyEval(Arrays.asList(vertices));
    }

    public static void lazyEval(Collection<? extends IVertex> vertices) {
        Deque<IVertex> stack = asDeque(vertices);

        while (!stack.isEmpty()) {

            IVertex<?> head = stack.peek();
            Set<IVertex<?>> parentsThatAreNotYetCalculated = parentsThatAreNotCalculated(head.getParents());

            if (((Vertex)head).isProbabilistic() || parentsThatAreNotYetCalculated.isEmpty()) {

                IVertex<?> top = stack.pop();
                top.updateValue();

            } else {

                for (IVertex<?> vertex : parentsThatAreNotYetCalculated) {
                    stack.push(vertex);
                }

            }

        }
    }

    private static Set<IVertex<?>> parentsThatAreNotCalculated(Set<? extends IVertex> parents) {
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
}
