package io.improbable.keanu.vertices.dbl;


import java.util.Map;
import java.util.function.Function;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcTanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TanVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class DoubleVertex extends Vertex<DoubleTensor> implements DoubleOperators<DoubleVertex>, Differentiable {

    public DoubleVertex(ValueUpdater<DoubleTensor> valueUpdater, Observable<DoubleTensor> observation) {
        super(valueUpdater, observation);
    }

    public DoubleVertex minus(DoubleVertex that) {
        return new DifferenceVertex(this, that);
    }

    public DoubleVertex plus(DoubleVertex that) {
        return new AdditionVertex(this, that);
    }

    public DoubleVertex multiply(DoubleVertex that) {
        return new MultiplicationVertex(this, that);
    }

    public DoubleVertex matrixMultiply(DoubleVertex that) {
        return new MatrixMultiplicationVertex(this, that);
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

    public DoubleVertex round() {
        return new RoundVertex(this);
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

    public DoubleVertex sum() {
        return new SumVertex(this);
    }

    public DoubleVertex lambda(int[] outputShape, Function<DoubleTensor, DoubleTensor> op, Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation) {
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

    public double getValue(int... index) {
        return getValue().getValue(index);
    }

    @Override
    public DualNumber calculateDualNumber(Map<IVertex, DualNumber> dualNumbers) {
        if (isObserved()) {
            return DualNumber.createConstant(getValue());
        } else {
            return DualNumber.createWithRespectToSelf(getId(), getValue());
        }
    }
}
