package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.*;

import static java.util.Collections.singleton;

public class Differentiator {

    public static PartialDerivatives forwardModeAutoDiff(DoubleVertex of, Collection<DoubleVertex> wrt) {
        return forwardModeAutoDiff(Collections.singletonList(of), wrt).get(of);
    }

    public static Map<Vertex, PartialDerivatives> forwardModeAutoDiff(Collection<DoubleVertex> of, Collection<DoubleVertex> wrt) {

        PriorityQueue<DoubleVertex> priorityQueue = new PriorityQueue<>(Comparator.comparingLong(Vertex::getId));
        priorityQueue.addAll(wrt);

        HashSet<Vertex> alreadyQueued = new HashSet<>(wrt);

        Map<Vertex, DualNumber> dualNumbers = new HashMap<>();
        Map<Vertex, PartialDerivatives> ofWrt = new HashMap<>();

        while (!priorityQueue.isEmpty()) {
            DoubleVertex visiting = priorityQueue.poll();

            DualNumber dualNumber = visiting.calculateDualNumber(dualNumbers);
            dualNumbers.put(visiting, dualNumber);
            if (of.contains(visiting)) {
                ofWrt.put(visiting, dualNumber.getPartialDerivatives());
            }

            for (Vertex child : visiting.getChildren()) {

                if (!child.isProbabilistic() && !alreadyQueued.contains(child) && child instanceof DoubleVertex) {
                    priorityQueue.offer((DoubleVertex) child);
                    alreadyQueued.add(child);
                }
            }
        }

        return ofWrt;
    }

    public static Map<Long, PartialDerivatives> reverseModeAutoDiff(Set<DoubleVertex> of, Set<DoubleVertex> wrt) {

        PriorityQueue<DoubleVertex> priorityQueue = new PriorityQueue<>(Comparator.<Vertex>comparingLong(Vertex::getId).reversed());
        priorityQueue.addAll(of);

        HashSet<Vertex> alreadyQueued = new HashSet<>(of);

        Map<Vertex, PartialDerivatives> dwrtOf = new HashMap<>();
        of.forEach(v -> dwrtOf.put(v, PartialDerivatives.withRespectToSelf(v.getId(), v.getShape())));

        Map<Long, PartialDerivatives> wrtOf = new HashMap<>();

        while (!priorityQueue.isEmpty()) {
            DoubleVertex visiting = priorityQueue.poll();

            if (wrt.contains(visiting)) {
                wrtOf.put(visiting.getId(), dwrtOf.get(visiting));
                continue;
            }

            Map<Vertex, PartialDerivatives> partialDerivatives = visiting.reverseModeAutoDifferentiation(dwrtOf.get(visiting));

            for (Map.Entry<Vertex, PartialDerivatives> v : partialDerivatives.entrySet()) {
                if (dwrtOf.containsKey(v.getKey())) {
                    dwrtOf.put(v.getKey(), v.getValue().add(dwrtOf.get(v.getKey())));
                } else {
                    dwrtOf.put(v.getKey(), v.getValue());
                }
            }

            if (!visiting.isProbabilistic()) {
                for (Vertex parent : visiting.getParents()) {
                    if (!alreadyQueued.contains(parent) && parent instanceof DoubleVertex) {
                        priorityQueue.offer((DoubleVertex) parent);
                        alreadyQueued.add(parent);
                    }
                }
            }
        }

        return wrtOfToOfWrt(wrtOf);
    }

    public static PartialDerivatives reverseModeAutoDiff(DoubleVertex of, Set<DoubleVertex> wrt) {
        return reverseModeAutoDiff(singleton(of), wrt).get(of.getId());
    }

    public static PartialDerivatives reverseModeAutoDiff(DoubleVertex of, DoubleVertex... wrt) {
        return reverseModeAutoDiff(singleton(of), new HashSet<>(Arrays.asList(wrt))).get(of.getId());
    }

    private static Map<Long, PartialDerivatives> wrtOfToOfWrt(Map<Long, PartialDerivatives> wrtOf) {
        Map<Long, PartialDerivatives> ofWrt = new HashMap<>();

        for (Map.Entry<Long, PartialDerivatives> wrtOfEntry : wrtOf.entrySet()) {
            Map<Long, DoubleTensor> ofs = wrtOfEntry.getValue().asMap();

            for (Map.Entry<Long, DoubleTensor> ofsEntry : ofs.entrySet()) {

                if (ofWrt.containsKey(ofsEntry.getKey())) {
                    ofWrt.get(ofsEntry.getKey()).putWithRespectTo(wrtOfEntry.getKey(), ofsEntry.getValue());
                } else {
                    ofWrt.put(ofsEntry.getKey(), new PartialDerivatives(wrtOfEntry.getKey(), ofsEntry.getValue()));
                }
            }
        }

        return ofWrt;
    }

    //TODO: move this into the PartialDerivative multiply
    public static PartialDerivatives reshapeReverseAutoDiff(PartialDerivatives partialDerivatives, DoubleTensor primary, DoubleTensor secondary) {
        Map<Long, DoubleTensor> reshapedPartials = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> partialDerivative : partialDerivatives.asMap().entrySet()) {
            DoubleTensor partial;
            if (primary.isScalar()) {
                int[] nonScalarDimensions = TensorShape.nonScalarDimensions(secondary.getShape());
                partial = partialDerivative.getValue().sum(nonScalarDimensions).reshape(TensorShape.concat(secondary.getShape(), primary.getShape()));
            } else {
                partial = partialDerivative.getValue();
            }
            reshapedPartials.put(partialDerivative.getKey(), partial);
        }

        return new PartialDerivatives(reshapedPartials);
    }

}
