package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import lombok.experimental.UtilityClass;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;
import java.util.function.Consumer;
import java.util.function.Predicate;

@UtilityClass
public class DifferentiableChecker {

    public boolean isDifferentiableWrtLatents(Collection<Vertex> vertices) {
        if (!allProbabilisticAreDoubleOrObserved(vertices)) {
            return false;
        }
        Set<Vertex> allParents = allParentsOf(vertices);
        Set<Vertex> constantValueVerticesCache = new HashSet<>();
        return diffableOrConstantUptoNextRV(allParents, constantValueVerticesCache);
    }

    private boolean allProbabilisticAreDoubleOrObserved(Collection<Vertex> vertices) {
        return vertices.stream().filter(Vertex::isProbabilistic)
            .allMatch(DifferentiableChecker::isDoubleOrObserved);
    }

    private boolean isDoubleOrObserved(Vertex v) {
        return (v instanceof DoubleVertex || v.isObserved());
    }

    private Set<Vertex> allParentsOf(Collection<Vertex> vertices) {
        Set<Vertex> allParents = new HashSet<>();
        for (Vertex vertex : vertices) {
            allParents.addAll(vertex.getParents());
        }
        return allParents;
    }

    private boolean diffableOrConstantUptoNextRV(Collection<Vertex> vertices, Set<Vertex> constantValueVerticesCache) {
        return bfsExplorer(vertices,
            vertex -> isNonDiffableAndNotConstant(vertex, constantValueVerticesCache),
            vertex -> !vertex.isProbabilistic(),
            doNothing -> {});
    }

    private boolean bfsExplorer(Collection<Vertex> vertices, Predicate<Vertex> failureCondition,
                                Predicate<Vertex> shouldAddParents,
                                Consumer<Collection<Vertex>> successfullyVisitedConsumer) {
        Queue<Vertex> queue = new LinkedList<>(vertices);
        Set<Vertex> queued = new HashSet<>(vertices);

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (failureCondition.test(visiting)) {
                return false;
            }

            if (shouldAddParents.test(visiting)) {
                Collection<Vertex> nextVertices = visiting.getParents();
                for (Vertex next : nextVertices) {
                    if (!queued.contains(next)) {
                        queue.offer(next);
                        queued.add(next);
                    }
                }
            }
        }
        successfullyVisitedConsumer.accept(queued);
        return true;
    }

    private boolean isNonDiffableAndNotConstant(Vertex vertex, Set<Vertex> constantValueVerticesCache) {
        return !vertex.isDifferentiable() && !isVertexValueConstant(Collections.singletonList(vertex), constantValueVerticesCache);
    }

    private boolean isVertexValueConstant(Collection<Vertex> vertices, Set<Vertex> constantValueVerticesCache) {
        return bfsExplorer(vertices,
            DifferentiableChecker::isUnobservedProbabilistic,
            vertex -> !isValueKnownToBeConstant(vertex, constantValueVerticesCache),
            constantValueVerticesCache::addAll);
    }

    private boolean isUnobservedProbabilistic(Vertex vertex) {
        return vertex.isProbabilistic() && !vertex.isObserved();
    }

    // We know whether these are constant. For cases such as a MultiplicationVertex we would need to
    // explore its parents to ensure its constant.
    private boolean isValueKnownToBeConstant(Vertex vertex, Set<Vertex> constantValueVerticesCache) {
        return vertex instanceof ConstantVertex || constantValueVerticesCache.contains(vertex) || vertex.isObserved();
    }
}
