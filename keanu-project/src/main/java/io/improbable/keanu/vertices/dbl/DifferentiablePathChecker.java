package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

public class DifferentiablePathChecker {

    private final Set<? extends Vertex<?>> wrtVertices;

    public DifferentiablePathChecker(Collection<? extends Vertex<?>> wrtVertices) {
        this.wrtVertices = new HashSet<>(wrtVertices);
    }

    public boolean differentiablePath(Collection<Vertex> vertices) {
        return vertices.stream().allMatch(this::differentiablePath);
    }

    public boolean differentiablePath(Vertex vertex) {
        if (isDiscreteLatent(vertex)) {
            return false;
        }
        Queue<Vertex> queue = new LinkedList<>();
        queue.offer(vertex);
        Set<Vertex> queued = new HashSet<>(Collections.singletonList(vertex));

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

    private boolean isDiscreteLatent(Vertex vertex) {
        boolean latent = vertex.isProbabilistic() && !vertex.isObserved();
        boolean discrete = !(vertex.getValue() instanceof DoubleTensor);
        return latent && discrete;
    }

    private boolean isConstantVertex(Vertex vertex) {
        Collection<Vertex> initialNext = vertex.getParents();
        Queue<Vertex> queue = new LinkedList<>(initialNext);
        Set<Vertex> queued = new HashSet<>(initialNext);

        while (!queue.isEmpty()) {
            Vertex visiting = queue.poll();

            if (visiting.isProbabilistic()) {
                if(visiting.isObserved()) {
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
