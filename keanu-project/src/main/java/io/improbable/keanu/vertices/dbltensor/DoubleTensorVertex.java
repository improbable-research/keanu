package io.improbable.keanu.vertices.dbltensor;


import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.ConstantTensorVertex;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff.TensorDualNumber;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.binary.*;
import io.improbable.keanu.vertices.dbltensor.nonprobabilistic.operators.unary.*;

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

    public DoubleTensorVertex divideBy(DoubleTensorVertex that) {
        return new TensorDivisionVertex(this, that);
    }

    public DoubleTensorVertex pow(DoubleTensorVertex exponent) {
        return new TensorPowerVertex(this, exponent);
    }

    public DoubleTensorVertex minus(double that) {
        return new TensorDifferenceVertex(this, new ConstantTensorVertex(that));
    }

    public DoubleTensorVertex plus(double that) {
        return new TensorAdditionVertex(this, new ConstantTensorVertex(that));
    }

    public DoubleTensorVertex multiply(double that) {
        return new TensorMultiplicationVertex(this, new ConstantTensorVertex(that));
    }

    public DoubleTensorVertex divideBy(double that) {
        return new TensorDivisionVertex(this, new ConstantTensorVertex(that));
    }

    public DoubleTensorVertex pow(double power) {
        return new TensorPowerVertex(this, new ConstantTensorVertex(power));
    }

    public DoubleTensorVertex abs() {
        return new TensorAbsVertex(this);
    }

    public DoubleTensorVertex floor() {
        return new TensorFloorVertex(this);
    }

    public DoubleTensorVertex ceil() {
        return new TensorCeilVertex(this);
    }

    public DoubleTensorVertex exp() {
        return new TensorExpVertex(this);
    }

    public DoubleTensorVertex log() {
        return new TensorLogVertex(this);
    }

    public DoubleTensorVertex sigmoid() {
        return new TensorSigmoidVertex(this);
    }

    public DoubleTensorVertex sin() {
        return new TensorSinVertex(this);
    }

    public DoubleTensorVertex cos() {
        return new TensorCosVertex(this);
    }

    public DoubleTensorVertex tan() {
        return new TensorTanVertex(this);
    }

    public DoubleTensorVertex asin() {
        return new TensorArcSinVertex(this);
    }

    public DoubleTensorVertex acos() {
        return new TensorArcCosVertex(this);
    }

    public DoubleTensorVertex atan() {
        return new TensorArcTanVertex(this);
    }

    public DoubleTensorVertex atan2(DoubleTensorVertex that) {
        return new TensorArcTan2Vertex(this, that);
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

    public void setValue(Double value) {
        super.setValue(DoubleTensor.create(value, getShape()));
    }

    public void setAndCascade(Double value) {
        super.setAndCascade(DoubleTensor.create(value, getShape()));
    }

    public void setAndCascade(Double value, Map<Long, Long> explored) {
        super.setAndCascade(DoubleTensor.create(value, getShape()), explored);
    }

    public void observe(Double value) {
        super.observe(DoubleTensor.create(value, getShape()));
    }

    public double logPdf(double value) {
        if (this.getValue().isScalar()) {
            return this.logPdf(DoubleTensor.scalar(value));
        } else {
            throw new IllegalArgumentException("Vertex is not scalar");
        }
    }

    public Map<Long, DoubleTensor> dLogPdf(double value) {
        if (this.getValue().isScalar()) {
            return this.dLogPdf(DoubleTensor.scalar(value));
        } else {
            throw new IllegalArgumentException("Vertex is not scalar");
        }
    }

}
