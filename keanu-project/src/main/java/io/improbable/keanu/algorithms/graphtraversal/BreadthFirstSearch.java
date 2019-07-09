package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.IVertex;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.HashSet;
import java.util.Queue;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;

public class BreadthFirstSearch {

    public static boolean bfsWithFailureCondition(Collection<IVertex> vertices,
                                                  Predicate<IVertex> failureCondition,
                                                  Function<IVertex, Collection<IVertex>> nextVertices,
                                                  Consumer<Collection<IVertex>> successfullyVisitedConsumer) {

        Queue<IVertex> queue = new ArrayDeque<>(vertices);
        Set<IVertex> visited = new HashSet<>(vertices);

        while (!queue.isEmpty()) {
            IVertex visiting = queue.poll();

            if (failureCondition.test(visiting)) {
                return false;
            }

            queueUnvisitedNextVertices(nextVertices.apply(visiting), queue, visited);
        }

        successfullyVisitedConsumer.accept(visited);
        return true;
    }

    private static void queueUnvisitedNextVertices(Collection<IVertex> nextVertices,
                                                   Queue<IVertex> queue,
                                                   Set<IVertex> visited) {

        for (IVertex next : nextVertices) {
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
