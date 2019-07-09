package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.Vertex;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.HashSet;
import java.util.Queue;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;

public class BreadthFirstSearch {

    public static boolean bfsWithFailureCondition(Collection<Vertex> vertices,
                                                  Predicate<Vertex> failureCondition,
                                                  Function<Vertex, Collection<Vertex>> nextVertices,
                                                  Consumer<Collection<Vertex>> successfullyVisitedConsumer) {

        Queue<Vertex> queue = new ArrayDeque<>(vertices);
        Set<Vertex> visited = new HashSet<>(vertices);

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (failureCondition.test(visiting)) {
                return false;
            }

            queueUnvisitedNextVertices(nextVertices.apply(visiting), queue, visited);
        }

        successfullyVisitedConsumer.accept(visited);
        return true;
    }

    private static void queueUnvisitedNextVertices(Collection<Vertex> nextVertices,
                                                   Queue<Vertex> queue,
                                                   Set<Vertex> visited) {

        for (Vertex next : nextVertices) {
            if (!visited.contains(next)) {
                queue.offer(next);
                visited.add(next);
            }
        }
    }

    public static <T> void doNothing(T toConsume) {
        return;
    }
}
