package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

@UtilityClass
public class DifferentiablePathChecker {

    public boolean differentiablePath(Collection<Vertex> vertices) {
        if (!vertices.stream().allMatch(Vertex::isDifferentiable)) {
            return false;
        }
        Queue<Vertex> queue = new LinkedList<>(vertices);
        Set<Vertex> queued = new HashSet<>(vertices);

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (!visiting.isDifferentiable() && !isConstantVertex(visiting)) {
                return false;
            }

            Collection<Vertex> nextVertices = visiting.getParents();
            for (Vertex next : nextVertices) {
                queue.offer(next);
                queued.add(next);
            }
        }
        return true;
    }

    public boolean differentiablePath(Vertex vertex) {
        return differentiablePath(Collections.singletonList(vertex));
    }

    private boolean isConstantVertex(Vertex vertex) {
        Collection<Vertex> initialNext = vertex.getParents();
        Queue<Vertex> queue = new LinkedList<>(initialNext);
        Set<Vertex> queued = new HashSet<>(initialNext);

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (visiting.isProbabilistic()) {
                if (visiting.isObserved()) {
                    continue;
                } else {
                    return false;
                }
            }

            Collection<Vertex> nextVertices = visiting.getParents();
            for (Vertex next : nextVertices) {
                if (!queued.contains(next)) {
                    queue.offer(next);
                    queued.add(next);
                }
            }
        }
        return false;
    }
}
