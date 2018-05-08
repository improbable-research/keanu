package io.improbable.keanu.vertices.dbltensor;


import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.TensorAdditionVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.TensorDifferenceVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.TensorDivisionVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.TensorMultiplicationVertex;

import java.util.*;

public abstract class DoubleTensorVertex extends ContinuousTensorVertex<DoubleTensor> {

    public DoubleTensorVertex minus(DoubleTensorVertex that) {
        return new TensorDifferenceVertex(this, that);
    }

    public DoubleTensorVertex plus(DoubleTensorVertex that) {
        return new TensorAdditionVertex(this, that);
    }

    public DoubleTensorVertex multiply(DoubleTensorVertex that) {
        return new TensorMultiplicationVertex(this, that);
    }

    public DoubleTensorVertex divideBy(DoubleTensorVertex that){
        return new TensorDivisionVertex(this, that);
    }

    public final TensorDualNumber getDualNumber() {
        Map<Vertex, TensorDualNumber> dualNumbers = new HashMap<>();
        Deque<DoubleTensorVertex> stack = new ArrayDeque<>();
        stack.push(this);

        while (!stack.isEmpty()) {

            DoubleTensorVertex head = stack.peek();
            Set<Vertex> parentsThatDualNumberIsNotCalculated = parentsThatDualNumberIsNotCalculated(dualNumbers, head.getParents());

            if (parentsThatDualNumberIsNotCalculated.isEmpty()) {

                DoubleTensorVertex top = stack.pop();
                TensorDualNumber dual = top.calculateDualNumber(dualNumbers);
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

    private Set<Vertex> parentsThatDualNumberIsNotCalculated(Map<Vertex, TensorDualNumber> dualNumbers, Set<Vertex> parents) {
        Set<Vertex> notCalculatedParents = new HashSet<>();
        for (Vertex<?> next : parents) {
            if (!dualNumbers.containsKey(next)) {
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }

    protected abstract TensorDualNumber calculateDualNumber(Map<Vertex, TensorDualNumber> dualNumbers);

}
