package io.improbable.keanu.vertices.dbltensor;


import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.MultiplicationVertex;

import java.util.*;

public abstract class DoubleTensorVertex extends ContinuousTensorVertex<DoubleTensor> {

    public DoubleTensorVertex minus(DoubleTensorVertex that) {
        return new DifferenceVertex(this, that);
    }

    public DoubleTensorVertex plus(DoubleTensorVertex that) {
        return new AdditionVertex(this, that);
    }

    public DoubleTensorVertex multiply(DoubleTensorVertex that) {
        return new MultiplicationVertex(this, that);
    }

    public final DualNumber getDualNumber() {
        Map<Vertex, DualNumber> dualNumbers = new HashMap<>();
        Deque<DoubleTensorVertex> stack = new ArrayDeque<>();
        stack.push(this);

        while (!stack.isEmpty()) {

            DoubleTensorVertex head = stack.peek();
            Set<Vertex> parentsThatDualNumberIsNotCalculated = parentsThatDualNumberIsNotCalculated(dualNumbers, head.getParents());

            if (parentsThatDualNumberIsNotCalculated.isEmpty()) {

                DoubleTensorVertex top = stack.pop();
                DualNumber dual = top.calculateDualNumber(dualNumbers);
                dualNumbers.put(top, dual);

            } else {

                for (Vertex vertex : parentsThatDualNumberIsNotCalculated) {
                    if (vertex instanceof DoubleTensorVertex) {
                        stack.push((DoubleTensorVertex) vertex);
                    } else {
                        throw new IllegalArgumentException("Can only calculate Dual Numbers on a graph made of Doubles");
                    }
                }

            }

        }

        return dualNumbers.get(this);
    }

    private Set<Vertex> parentsThatDualNumberIsNotCalculated(Map<Vertex, DualNumber> dualNumbers, Set<Vertex> parents) {
        Set<Vertex> notCalculatedParents = new HashSet<>();
        for (Vertex<?> next : parents) {
            if (!dualNumbers.containsKey(next)) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }

    protected abstract DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers);

}
