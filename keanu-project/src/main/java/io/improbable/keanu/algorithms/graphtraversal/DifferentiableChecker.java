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

@UtilityClass
public class DifferentiableChecker {

    public boolean isDifferentiable(Collection<Vertex> vertices) {
        if(!allProbabilisticAreDoubleOrObserved(vertices)) {
            return false;
        }
        Set<Vertex> allParents = getSetOfAllParents(vertices);
        return diffableOrConstantUptoNextRV(allParents);
    }
    private boolean allProbabilisticAreDoubleOrObserved(Collection<Vertex> vertices) {
        return vertices.stream().filter(Vertex::isProbabilistic)
            .allMatch(DifferentiableChecker::isDoubleOrObserved);
    }

    private boolean isDoubleOrObserved(Vertex v) {
        return (v instanceof DoubleVertex || v.isObserved());
    }

    public Set<Vertex> getSetOfAllParents(Collection<Vertex> vertices) {
        Set<Vertex> allParents = new HashSet<>();
        for (Vertex vertex : vertices) {
            allParents.addAll(vertex.getParents());
        }
        return allParents;
    }

    private boolean diffableOrConstantUptoNextRV(Collection<Vertex> vertices) {
        Queue<Vertex> queue = new LinkedList<>(vertices);
        Set<Vertex> queued = new HashSet<>(vertices);
        Set<Vertex> constantValueVerticesCache = new HashSet<>();

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (isNonDiffableAndNotConstant(visiting, constantValueVerticesCache)) {
                return false;
            }

            if (!visiting.isProbabilistic()) {
                Collection<Vertex> nextVertices = visiting.getParents();
                for (Vertex next : nextVertices) {
                    if (!queued.contains(next)) {
                        queue.offer(next);
                        queued.add(next);
                    }
                }
            }
        }
        return true;
    }

    private boolean isNonDiffableAndNotConstant(Vertex vertex, Set<Vertex> constantValueVerticesCache) {
        return !vertex.isDifferentiable() && !isVertexValueConstant(vertex, constantValueVerticesCache);
    }

    private boolean isVertexValueConstant(Vertex vertex, Set<Vertex> constantValueVerticesCache) {
        Collection<Vertex> initialNext = Collections.singletonList(vertex);
        Queue<Vertex> queue = new LinkedList<>(initialNext);
        Set<Vertex> queued = new HashSet<>(initialNext);

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (isUnobservedProbabilistic(visiting)) {
                return false;
            }

            if (!isValueKnownToBeConstant(visiting, constantValueVerticesCache)) {
                Collection<Vertex> nextVertices = visiting.getParents();
                for (Vertex next : nextVertices) {
                    if (!queued.contains(next)) {
                        queue.offer(next);
                        queued.add(next);
                    }
                }
            }
        }
        constantValueVerticesCache.addAll(queued);
        return true;
    }

    // We know whether these are constant. For cases such as a MultiplicationVertex we would need to
    // explore its parents to ensure its constant.
    private boolean isValueKnownToBeConstant(Vertex vertex, Set<Vertex> constantValueVerticesCache) {
        return vertex instanceof ConstantVertex || constantValueVerticesCache.contains(vertex) || vertex.isObserved();
    }

    private boolean isUnobservedProbabilistic(Vertex vertex) {
        return vertex.isProbabilistic() && !vertex.isObserved();
    }
}
