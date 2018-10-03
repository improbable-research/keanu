package io.improbable.keanu.vertices.dbl;

import static java.util.Collections.singletonMap;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;

public class Differentiator {

    public static PartialDerivatives reverseModeAutoDiff(
            Vertex<?> ofVertex, PartialDerivatives dWrtOfVertex, Set<? extends Vertex<?>> wrt) {

        PriorityQueue<Vertex> priorityQueue =
                new PriorityQueue<>(
                        Comparator.<Vertex, VertexId>comparing(
                                        Vertex::getId, Comparator.naturalOrder())
                                .reversed());
        priorityQueue.add(ofVertex);

        HashSet<Vertex> alreadyQueued = new HashSet<>();
        alreadyQueued.add(ofVertex);

        Map<Vertex, PartialDerivatives> dwrtOf = new HashMap<>();
        collectPartials(singletonMap(ofVertex, dWrtOfVertex), dwrtOf, ofVertex);

        Map<VertexId, PartialDerivatives> wrtOf = new HashMap<>();

        Vertex<?> visiting;
        while ((visiting = priorityQueue.poll()) != null) {

            if (wrt.contains(visiting)) {
                wrtOf.put(visiting.getId(), dwrtOf.get(visiting));
                continue;
            }

            if (visiting instanceof Differentiable) {
                Differentiable visitingDifferentiable = ((Differentiable) visiting);
                Map<Vertex, PartialDerivatives> partialDerivatives =
                        visitingDifferentiable.reverseModeAutoDifferentiation(dwrtOf.get(visiting));
                collectPartials(partialDerivatives, dwrtOf, visiting);
            }

            if (!visiting.isProbabilistic()) {
                for (Vertex parent : visiting.getParents()) {
                    if (!alreadyQueued.contains(parent) && parent instanceof Differentiable) {
                        priorityQueue.offer(parent);
                        alreadyQueued.add(parent);
                    }
                }
            }
        }

        return wrtOfToOfWrt(wrtOf).get(ofVertex.getId());
    }

    private static void collectPartials(
            Map<Vertex, PartialDerivatives> partialDerivatives,
            Map<Vertex, PartialDerivatives> dwrtOf,
            Vertex visiting) {

        for (Map.Entry<Vertex, PartialDerivatives> v : partialDerivatives.entrySet()) {

            Vertex wrtVertex = v.getKey();
            PartialDerivatives partialsOf = v.getValue();
            int[] wrtShape = wrtVertex.getShape();
            int prevRank = visiting.getShape().length;

            PartialDerivatives dwrtV;
            if (TensorShape.isScalar(wrtShape)) {
                dwrtV = partialsOf.sumOverWrtDimensions(TensorShape.dimensionRange(-prevRank, 0));
            } else {
                dwrtV = partialsOf;
            }

            if (dwrtOf.containsKey(wrtVertex)) {
                dwrtOf.put(wrtVertex, dwrtOf.get(wrtVertex).add(dwrtV));
            } else {
                dwrtOf.put(wrtVertex, dwrtV);
            }
        }
    }

    public static PartialDerivatives reverseModeAutoDiff(Vertex ofVertex, Set<DoubleVertex> wrt) {
        return reverseModeAutoDiff(
                ofVertex,
                PartialDerivatives.withRespectToSelf(ofVertex.getId(), ofVertex.getShape()),
                wrt);
    }

    public static PartialDerivatives reverseModeAutoDiff(Vertex ofVertex, DoubleVertex... wrt) {
        return reverseModeAutoDiff(ofVertex, new HashSet<>(Arrays.asList(wrt)));
    }

    /**
     * Reorganize collection of partials to be easily used to get partial OF Y WRT X. This structure
     * is what forward mode auto diff returns but needs to be used on reverse mode so that it is in
     * the same form.
     *
     * @param wrtOf map of partials where key is wrt vertex and key in partial is key of vertex
     * @return a reordered map with the key being the of vertex and the key in the partial being wrt
     *     vertex
     */
    private static Map<VertexId, PartialDerivatives> wrtOfToOfWrt(
            Map<VertexId, PartialDerivatives> wrtOf) {
        Map<VertexId, PartialDerivatives> ofWrt = new HashMap<>();

        for (Map.Entry<VertexId, PartialDerivatives> wrtOfEntry : wrtOf.entrySet()) {
            Map<VertexId, DoubleTensor> ofs = wrtOfEntry.getValue().asMap();

            for (Map.Entry<VertexId, DoubleTensor> ofsEntry : ofs.entrySet()) {

                if (ofWrt.containsKey(ofsEntry.getKey())) {
                    ofWrt.get(ofsEntry.getKey())
                            .putWithRespectTo(wrtOfEntry.getKey(), ofsEntry.getValue());
                } else {
                    ofWrt.put(
                            ofsEntry.getKey(),
                            new PartialDerivatives(wrtOfEntry.getKey(), ofsEntry.getValue()));
                }
            }
        }

        return ofWrt;
    }

    public static <V extends Vertex & Differentiable> DualNumber calculateDual(V vertex) {
        Map<Vertex, DualNumber> dualNumbers = new HashMap<>();
        Deque<V> stack = new ArrayDeque<>();
        stack.push(vertex);

        while (!stack.isEmpty()) {

            V head = stack.peek();
            Set<Vertex> parentsThatDualNumberIsNotCalculated =
                    parentsThatDualNumberIsNotCalculated(dualNumbers, head.getParents());

            if (parentsThatDualNumberIsNotCalculated.isEmpty()) {
                V top = stack.pop();
                DualNumber dual = top.calculateDualNumber(dualNumbers);
                dualNumbers.put(top, dual);
            } else {
                for (Vertex parent : parentsThatDualNumberIsNotCalculated) {
                    if (parent instanceof Differentiable) {
                        stack.push((V) parent);
                    }
                }
            }
        }

        return dualNumbers.get(vertex);
    }

    private static Set<Vertex> parentsThatDualNumberIsNotCalculated(
            Map<Vertex, DualNumber> dualNumbers, Collection<? extends Vertex> parents) {
        Set<Vertex> notCalculatedParents = new HashSet<>();
        for (Vertex next : parents) {
            if (!dualNumbers.containsKey(next) && next instanceof Differentiable) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }
}
