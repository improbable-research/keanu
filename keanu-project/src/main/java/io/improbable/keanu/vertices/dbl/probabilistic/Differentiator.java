package io.improbable.keanu.vertices.dbl.probabilistic;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

public class Differentiator {
    public DualNumber calculateDual(Differentiable vertex) {
        Map<IVertex, DualNumber> dualNumbers = new HashMap<>();
        Deque<Differentiable> stack = new ArrayDeque<>();
        stack.push(vertex);

        while (!stack.isEmpty()) {

            Differentiable head = stack.peek();
            Set<IVertex> parentsThatDualNumberIsNotCalculated = parentsThatDualNumberIsNotCalculated(dualNumbers, head.getParents());

            if (parentsThatDualNumberIsNotCalculated.isEmpty()) {

                Differentiable top = stack.pop();
                DualNumber dual = top.calculateDualNumber(dualNumbers);
                dualNumbers.put(top, dual);

            } else {

                for (IVertex parent : parentsThatDualNumberIsNotCalculated) {
                    if (parent instanceof Differentiable) {
                        stack.push((Differentiable) parent);
                    } else {
                        throw new IllegalArgumentException("Can only calculate Dual Numbers on a graph made of Differentiable vertices");
                    }
                }

            }

        }

        return dualNumbers.get(vertex);
    }

    private Set<IVertex> parentsThatDualNumberIsNotCalculated(Map<IVertex, DualNumber> dualNumbers, Set<? extends IVertex> parents) {
        Set<IVertex> notCalculatedParents = new HashSet<>();
        for (IVertex next : parents) {
            if (!dualNumbers.containsKey(next)) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }
}
