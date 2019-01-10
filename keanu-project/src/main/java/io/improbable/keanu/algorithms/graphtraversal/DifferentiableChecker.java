package io.improbable.keanu.algorithms.graphtraversal;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

@UtilityClass
public class DifferentiableChecker {

    /**
     * Checks whether a set of vertices are differentiable
     * w.r.t the latents in a graph.
     *
     * @param vertices the collection of vertices to check
     * @return true if the vertices are differentiable w.r.t latents
     */
    public boolean isDifferentiable(Collection<Vertex> vertices) {
        Queue<Vertex> queue = new LinkedList<>(vertices);
        Set<Vertex> queued = new HashSet<>(vertices);
        Set<Vertex> constantValueVerticesCache = new HashSet<>();

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (isNonDiffableAndNotConstant(visiting, constantValueVerticesCache)) {
                return false;
            }

            if (visiting.isDifferentiable()) {
                Collection<Vertex> nextVertices = visiting.getParents();
                for (Vertex next : nextVertices) {
                    if (!queued.contains(next) && !isValueKnownToBeConstant(next, constantValueVerticesCache)) {
                        queue.offer(next);
                        queued.add(next);
                    }
                }
            }
        }
        return true;
    }

    private boolean isNonDiffableAndNotConstant(Vertex vertex, Set<Vertex> constantValueVerticesCache) {
        return !vertex.isDifferentiable() && !isVertexConstant(vertex, constantValueVerticesCache);
    }

    private boolean isVertexConstant(Vertex vertex, Set<Vertex> constantValueVerticesCache) {
        if (vertex.isProbabilistic() && !vertex.isObserved()) {
            return false;
        }
        if (!isVertexParentsValueConstant(vertex, constantValueVerticesCache)) {
            return false;
        }
        return true;
    }

    private boolean isVertexParentsValueConstant(Vertex vertex, Set<Vertex> constantValueVerticesCache) {
        Collection<Vertex> initialNext = vertex.getParents();
        Queue<Vertex> queue = new LinkedList<>(initialNext);
        Set<Vertex> queued = new HashSet<>(initialNext);

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (isUnobservedProbabilistic(visiting)) {
                return false;
            }

            if (!isValueKnownToBeConstant(vertex, constantValueVerticesCache)) {
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
