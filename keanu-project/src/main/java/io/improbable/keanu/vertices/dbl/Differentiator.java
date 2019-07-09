package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsOf;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialsWithRespectTo;
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

    public static <V extends IVertex & Differentiable> PartialsWithRespectTo forwardModeAutoDiff(V wrt, V... of) {
        return forwardModeAutoDiff(wrt, new HashSet<>(Arrays.asList(of)));
    }

    public static <V extends IVertex & Differentiable> PartialsWithRespectTo forwardModeAutoDiff(V wrt, Collection<V> of) {

        PriorityQueue<V> priorityQueue = new PriorityQueue<>(Comparator.comparing(IVertex::getId, Comparator.naturalOrder()));
        priorityQueue.add(wrt);

        HashSet<IVertex> alreadyQueued = new HashSet<>();
        alreadyQueued.add(wrt);

        Map<IVertex, PartialDerivative> partials = new HashMap<>();
        Map<VertexId, PartialDerivative> ofWrt = new HashMap<>();

        while (!priorityQueue.isEmpty()) {
            V visiting = priorityQueue.poll();

            PartialDerivative partialOfVisiting = visiting.forwardModeAutoDifferentiation(partials);
            partials.put(visiting, partialOfVisiting);

            if (of.contains(visiting)) {
                ofWrt.put(visiting.getId(), partialOfVisiting);
                continue;
            }

            for (IVertex child : (Set<IVertex<?>>) visiting.getChildren()) {
                if (!child.isProbabilistic() && !alreadyQueued.contains(child) && child.isDifferentiable()) {
                    priorityQueue.offer((V) child);
                    alreadyQueued.add(child);
                }
            }
        }

        return new PartialsWithRespectTo(wrt, ofWrt);
    }

    public static PartialsOf reverseModeAutoDiff(IVertex ofVertex, Set<? extends IVertex<?>> wrt) {
        if (ofVertex.isObserved()) {
            return new PartialsOf(ofVertex, Collections.emptyMap());
        } else {
            return reverseModeAutoDiff(ofVertex, Differentiable.withRespectToSelf(ofVertex.getShape()), wrt);
        }
    }

    public static PartialsOf reverseModeAutoDiff(IVertex ofVertex, IVertex<?>... wrt) {
        return reverseModeAutoDiff(ofVertex, new HashSet<>(Arrays.asList(wrt)));
    }

    public static PartialsOf reverseModeAutoDiff(IVertex<?> ofVertex, PartialDerivative dWrtOfVertex, Set<? extends IVertex<?>> wrt) {

        ensureGraphValuesAndShapesAreSet(ofVertex);

        PriorityQueue<IVertex> priorityQueue = new PriorityQueue<>(Comparator.<IVertex, VertexId>comparing(IVertex::getId, Comparator.naturalOrder()).reversed());
        priorityQueue.add(ofVertex);

        HashSet<IVertex> alreadyQueued = new HashSet<>();
        alreadyQueued.add(ofVertex);

        Map<IVertex, PartialDerivative> dwrtOf = new HashMap<>();
        dwrtOf.put(ofVertex, dWrtOfVertex);

        Map<VertexId, PartialDerivative> wrtOf = new HashMap<>();

        IVertex<?> visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (wrt.contains(visiting)) {
                wrtOf.put(visiting.getId(), dwrtOf.get(visiting));
                continue;
            }

            if (!visiting.isProbabilistic()) {

                if (visiting.isDifferentiable()) {

                    Differentiable visitingDifferentiable = ((Differentiable) visiting);
                    PartialDerivative derivativeOfOutputWrtVisiting = dwrtOf.get(visiting);

                    if (derivativeOfOutputWrtVisiting != null) {

                        Map<IVertex, PartialDerivative> partialDerivatives = visitingDifferentiable.reverseModeAutoDifferentiation(derivativeOfOutputWrtVisiting);
                        collectPartials(partialDerivatives, dwrtOf);

                        for (IVertex parent : visiting.getParents()) {
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

    private static void ensureGraphValuesAndShapesAreSet(IVertex<?> vertex) {
        vertex.getValue();
    }

    private static void collectPartials(Map<IVertex, PartialDerivative> partialDerivatives,
                                        Map<IVertex, PartialDerivative> dwrtOf) {

        for (Map.Entry<IVertex, PartialDerivative> v : partialDerivatives.entrySet()) {

            IVertex wrtVertex = v.getKey();
            PartialDerivative dwrtV = v.getValue();

            if (dwrtOf.containsKey(wrtVertex)) {
                dwrtOf.put(wrtVertex, dwrtOf.get(wrtVertex).add(dwrtV));
            } else {
                dwrtOf.put(wrtVertex, dwrtV);
            }
        }
    }
}
