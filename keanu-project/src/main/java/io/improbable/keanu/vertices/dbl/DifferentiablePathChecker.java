package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.Vertex;
import lombok.experimental.UtilityClass;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;
import java.util.Set;

@UtilityClass
public class DifferentiablePathChecker {

    public boolean differentiablePath(Collection<Vertex> vertices) {
        if (!vertices.stream().allMatch(Vertex::isDifferentiable)) {
            return false;
        }
        Map<Vertex, Boolean> cache = new HashMap<>();
        for (Vertex v : vertices) {
            if (!differentiablePathWithCache(v, cache)) {
                return false;
            }
        }
        return true;
    }

    public boolean differentiablePath(Vertex vertex) {
        return differentiablePath(Collections.singletonList(vertex));
    }

    private boolean differentiablePathWithCache(Vertex vertex, Map<Vertex, Boolean> cachedResults) {
        Queue<Vertex> queue = new LinkedList<>(Collections.singletonList(vertex));
        Set<Vertex> queued = new HashSet<>(Collections.singletonList(vertex));

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (visiting.isObserved()) {
                continue;
            }

            if (cachedResults.containsKey(visiting)) {
                if (cachedResults.get(visiting)) {
                    continue;
                } else {
                    return false;
                }
            }

            if (!visiting.isDifferentiable()) {
                if (isVertexParentsConstant(visiting)) {
                    cachedResults.put(visiting, true);
                    continue;
                } else {
                    cachedResults.put(visiting, false);
                    return false;
                }
            }

            Collection<Vertex> nextVertices = visiting.getParents();
            for (Vertex next : nextVertices) {
                queue.offer(next);
                queued.add(next);
            }
        }
        cachedResults.put(vertex, true);
        return true;
    }

    private boolean isVertexParentsConstant(Vertex vertex) {
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
