package io.improbable.keanu.vertices.dbl;


import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ContinuousVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.*;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.*;

import java.util.*;
import java.util.function.Function;

public abstract class DoubleVertex extends ContinuousVertex<DoubleTensor> implements DoubleOperators<DoubleVertex> {

    public DoubleVertex minus(DoubleVertex that) {
        return new DifferenceVertex(this, that);
    }

    public DoubleVertex plus(DoubleVertex that) {
        return new AdditionVertex(this, that);
    }

    public DoubleVertex multiply(DoubleVertex that) {
        return new MultiplicationVertex(this, that);
    }

    public DoubleVertex divideBy(DoubleVertex that) {
        return new DivisionVertex(this, that);
    }

    public DoubleVertex pow(DoubleVertex exponent) {
        return new PowerVertex(this, exponent);
    }

    public DoubleVertex minus(double that) {
        return new DifferenceVertex(this, new ConstantDoubleVertex(that));
    }

    public DoubleVertex plus(double that) {
        return new AdditionVertex(this, new ConstantDoubleVertex(that));
    }

    public DoubleVertex multiply(double that) {
        return new MultiplicationVertex(this, new ConstantDoubleVertex(that));
    }

    public DoubleVertex divideBy(double that) {
        return new DivisionVertex(this, new ConstantDoubleVertex(that));
    }

    public DoubleVertex pow(double power) {
        return new PowerVertex(this, new ConstantDoubleVertex(power));
    }

    public DoubleVertex abs() {
        return new AbsVertex(this);
    }

    public DoubleVertex floor() {
        return new FloorVertex(this);
    }

    public DoubleVertex ceil() {
        return new CeilVertex(this);
    }

    public DoubleVertex exp() {
        return new ExpVertex(this);
    }

    public DoubleVertex log() {
        return new LogVertex(this);
    }

    public DoubleVertex sigmoid() {
        return new SigmoidVertex(this);
    }

    public DoubleVertex sin() {
        return new SinVertex(this);
    }

    public DoubleVertex cos() {
        return new CosVertex(this);
    }

    public DoubleVertex tan() {
        return new TanVertex(this);
    }

    public DoubleVertex asin() {
        return new ArcSinVertex(this);
    }

    public DoubleVertex acos() {
        return new ArcCosVertex(this);
    }

    public DoubleVertex atan() {
        return new ArcTanVertex(this);
    }

    public DoubleVertex atan2(DoubleVertex that) {
        return new ArcTan2Vertex(this, that);
    }

    public DoubleVertex lambda(int[] outputShape, Function<DoubleTensor, DoubleTensor> op, Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation) {
        return new DoubleUnaryOpLambda<>(outputShape, this, op, dualNumberCalculation);
    }

    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)
    public DoubleVertex times(DoubleVertex that) {
        return multiply(that);
    }

    public DoubleVertex div(DoubleVertex that) {
        return divideBy(that);
    }

    public DoubleVertex times(double that) {
        return multiply(that);
    }

    public DoubleVertex div(double that) {
        return divideBy(that);
    }

    public DoubleVertex unaryMinus() {
        return multiply(-1.0);
    }

    public final DualNumber getDualNumber() {
        Map<Vertex, DualNumber> dualNumbers = new HashMap<>();
        Deque<DoubleVertex> stack = new ArrayDeque<>();
        stack.push(this);

        while (!stack.isEmpty()) {

            DoubleVertex head = stack.peek();
            Set<Vertex> parentsThatDualNumberIsNotCalculated = parentsThatDualNumberIsNotCalculated(dualNumbers, head.getParents());

            if (parentsThatDualNumberIsNotCalculated.isEmpty()) {

                DoubleVertex top = stack.pop();
                DualNumber dual = top.calculateDualNumber(dualNumbers);
                dualNumbers.put(top, dual);

            } else {

                for (Vertex vertex : parentsThatDualNumberIsNotCalculated) {
                    if (vertex instanceof DoubleVertex) {
                        stack.push((DoubleVertex) vertex);
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

    public void setValue(double value) {
        super.setValue(DoubleTensor.create(value, getShape()));
    }

    public void setValue(double[] values) {
        super.setValue(DoubleTensor.create(values, getShape()));
    }

    public void setAndCascade(double value) {
        super.setAndCascade(DoubleTensor.create(value, getShape()));
    }

    public void setAndCascade(double[] values) {
        super.setAndCascade(DoubleTensor.create(values, getShape()));
    }

    public void setAndCascade(double value, Map<Long, Long> explored) {
        super.setAndCascade(DoubleTensor.create(value, getShape()), explored);
    }

    public void setAndCascade(double[] values, Map<Long, Long> explored) {
        super.setAndCascade(DoubleTensor.create(values, getShape()), explored);
    }

    public void observe(double value) {
        super.observe(DoubleTensor.create(value, getShape()));
    }

    public void observe(double[] values) {
        super.observe(DoubleTensor.create(values, getShape()));
    }

    public double logPdf(double value) {
        return this.logPdf(DoubleTensor.scalar(value));
    }

    public double logPdf(double[] values) {
        return this.logPdf(DoubleTensor.create(values));
    }

    public Map<Long, DoubleTensor> dLogPdf(double value) {
        return this.dLogPdf(DoubleTensor.scalar(value));
    }

    public Map<Long, DoubleTensor> dLogPdf(double[] values) {
        return this.dLogPdf(DoubleTensor.create(values));
    }
}
