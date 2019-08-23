package io.improbable.keanu.vertices.tensor.number.floating.dbl;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialsWithRespectTo;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;
import lombok.experimental.UtilityClass;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;


@UtilityClass
public class Differentiator {

    public static <V extends Vertex> PartialsWithRespectTo forwardModeAutoDiff(V wrt, V... of) {
        return forwardModeAutoDiff(wrt, new HashSet<>(Arrays.asList(of)));
    }

    public static <V extends Vertex> PartialsWithRespectTo forwardModeAutoDiff(V wrt, Collection<V> of) {

        PriorityQueue<V> priorityQueue = new PriorityQueue<>(Comparator.comparing(Vertex::getId, Comparator.naturalOrder()));
        priorityQueue.add(wrt);

        HashSet<Vertex> alreadyQueued = new HashSet<>();
        alreadyQueued.add(wrt);

        Map<Vertex, ForwardModePartialDerivative> partials = new HashMap<>();
        Map<VertexId, ForwardModePartialDerivative> ofWrt = new HashMap<>();

        while (!priorityQueue.isEmpty()) {
            V visiting = priorityQueue.poll();

            ForwardModePartialDerivative partialOfVisiting = ((Differentiable) visiting).forwardModeAutoDifferentiation(partials);
            partials.put(visiting, partialOfVisiting);

            if (of.contains(visiting)) {
                ofWrt.put(visiting.getId(), partialOfVisiting);
                continue;
            }

            for (Vertex child : (Set<Vertex>) visiting.getChildren()) {
                if (!child.isProbabilistic() && !alreadyQueued.contains(child) && child.isDifferentiable()) {
                    priorityQueue.offer((V) child);
                    alreadyQueued.add(child);
                }
            }
        }

        return new PartialsWithRespectTo(wrt, ofWrt);
    }

    public static PartialsOf reverseModeAutoDiff(Vertex ofVertex, Set<? extends Vertex> wrt) {
        if (ofVertex.isObserved()) {
            return new PartialsOf(ofVertex, Collections.emptyMap());
        } else {
            return reverseModeAutoDiff(ofVertex, Differentiable.ofSelfWrtSelf(ofVertex.getShape()), wrt);
        }
    }

    public static PartialsOf reverseModeAutoDiff(Vertex ofVertex, Vertex... wrt) {
        return reverseModeAutoDiff(ofVertex, new HashSet<>(Arrays.asList(wrt)));
    }

    public static PartialsOf reverseModeAutoDiff(Vertex ofVertex, ReverseModePartialDerivative dWrtOfVertex, Set<? extends Vertex> wrt) {

        ensureGraphValuesAndShapesAreSet(ofVertex);

        PriorityQueue<Vertex> priorityQueue = new PriorityQueue<>(Comparator.<Vertex, VertexId>comparing(Vertex::getId, Comparator.naturalOrder()).reversed());
        priorityQueue.add(ofVertex);

        HashSet<Vertex> alreadyQueued = new HashSet<>();
        alreadyQueued.add(ofVertex);

        Map<Vertex, ReverseModePartialDerivative> dwrtOf = new HashMap<>();
        dwrtOf.put(ofVertex, dWrtOfVertex);

        Map<VertexId, ReverseModePartialDerivative> wrtOf = new HashMap<>();

        Vertex<?, ?> visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (wrt.contains(visiting)) {
                wrtOf.put(visiting.getId(), dwrtOf.get(visiting));
                continue;
            }

            if (!visiting.isProbabilistic()) {

                if (visiting.isDifferentiable()) {

                    Differentiable visitingDifferentiable = ((Differentiable) visiting);
                    ReverseModePartialDerivative derivativeOfOutputWrtVisiting = dwrtOf.get(visiting);

                    if (derivativeOfOutputWrtVisiting != null) {

                        Map<Vertex, ReverseModePartialDerivative> partialDerivatives = visitingDifferentiable.reverseModeAutoDifferentiation(derivativeOfOutputWrtVisiting);
                        collectPartials(partialDerivatives, dwrtOf);

                        for (Vertex parent : visiting.getParents()) {
                            if (!alreadyQueued.contains(parent) && parent.isDifferentiable()) {
                                priorityQueue.offer(parent);
                                alreadyQueued.add(parent);
                            }
                        }

                    }
                }

            }
        }

        return new PartialsOf(ofVertex, wrtOf);
    }

    private static void ensureGraphValuesAndShapesAreSet(Vertex<?, ?> vertex) {
        vertex.getValue();
    }

    private static void collectPartials(Map<Vertex, ReverseModePartialDerivative> partialDerivatives,
                                        Map<Vertex, ReverseModePartialDerivative> dwrtOf) {

        for (Map.Entry<Vertex, ReverseModePartialDerivative> v : partialDerivatives.entrySet()) {

            Vertex wrtVertex = v.getKey();
            ReverseModePartialDerivative dwrtV = v.getValue();

            if (dwrtOf.containsKey(wrtVertex)) {
                dwrtOf.put(wrtVertex, dwrtOf.get(wrtVertex).add(dwrtV));
            } else {
                dwrtOf.put(wrtVertex, dwrtV);
            }
        }
    }
}
