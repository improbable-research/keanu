package io.improbable.keanu.vertices.dbl;


import java.util.Map;
import java.util.function.Function;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.BooleanBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleBinaryOpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiable;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class DoubleVertex extends Vertex<DoubleTensor> implements DoubleOperators<DoubleVertex>, Differentiable {

    public DoubleVertex(ValueUpdater<DoubleTensor> valueUpdater, Observable<DoubleTensor> observation) {
        super(valueUpdater, observation);
    }

    @Override
    public DoubleVertex minus(DoubleVertex that) {
        return new DoubleBinaryOpVertex(this, that,  (a, b) -> a.minus(b), (a, b) -> a.minus(b));
    }

    @Override
    public DoubleVertex plus(DoubleVertex that) {
        return new DoubleBinaryOpVertex(this, that,  (a, b) -> a.plus(b), (a, b) -> a.plus(b));
    }

    public DoubleVertex multiply(DoubleVertex that) {
        return new DoubleBinaryOpVertex(this, that,  (a, b) -> a.times(b), (a, b) -> a.times(b));
    }

    public DoubleVertex matrixMultiply(DoubleVertex that) {
        return new MatrixMultiplicationVertex(this, that);
    }

    public DoubleVertex divideBy(DoubleVertex that) {
        return new DoubleBinaryOpVertex(this, that,  (a, b) -> a.div(b), (a, b) -> a.div(b));
    }

    @Override
    public DoubleVertex pow(DoubleVertex that) {
        return new DoubleBinaryOpVertex(this, that,  (a, b) -> a.pow(b), (a, b) -> a.pow(b));
    }

    @Override
    public DoubleVertex minus(double that) {
        return minus(new ConstantDoubleVertex(that));
    }

    @Override
    public DoubleVertex plus(double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    public DoubleVertex multiply(double that) {
        return multiply(new ConstantDoubleVertex(that));
    }

    public DoubleVertex divideBy(double that) {
        return divideBy(new ConstantDoubleVertex(that));
    }

    @Override
    public DoubleVertex pow(double that) {
        return pow(new ConstantDoubleVertex(that));
    }

    public DoubleVertex abs() {
        return new DoubleUnaryOpVertex(this, a -> a.abs());
    }

    public DoubleVertex floor() {
        return new DoubleUnaryOpVertex(this, a -> a.floor());
    }

    public DoubleVertex ceil() {
        return new DoubleUnaryOpVertex( this, a -> a.ceil());
    }

    public DoubleVertex round() {
        return new DoubleUnaryOpVertex( this, a -> a.round());
    }

    @Override
    public DoubleVertex exp() {
        return new DoubleUnaryOpVertex(this, a -> a.exp(), a -> a.exp());
    }

    @Override
    public DoubleVertex log() {
        return new DoubleUnaryOpVertex(this, a -> a.log(), a -> a.log());
    }

    public DoubleVertex sigmoid() {
        return new DoubleUnaryOpVertex(this, a -> a.unaryMinus().expInPlace().plusInPlace(1).reciprocalInPlace(), a -> {
            DoubleTensor x = a.getValue();
            DoubleTensor xExp = x.exp();
            DoubleTensor dxdfx = xExp.divInPlace(xExp.plus(1).powInPlace(2));
            PartialDerivatives infinitesimal = a.getPartialDerivatives().multiplyBy(dxdfx);
            return new DualNumber(x.sigmoid(), infinitesimal);
        });
    }

    @Override
    public DoubleVertex sin() {
        return new DoubleUnaryOpVertex(this, a -> a.sin(), a -> a.sin());
    }

    @Override
    public DoubleVertex cos() {
        return new DoubleUnaryOpVertex(this, a -> a.cos(), a -> a.cos());
    }

    public DoubleVertex tan() {
        return new DoubleUnaryOpVertex(this, a -> a.tan(), a -> a.tan());
    }

    @Override
    public DoubleVertex asin() {
        return new DoubleUnaryOpVertex(this, a -> a.asin(), a -> a.asin());
    }

    @Override
    public DoubleVertex acos() {
        return new DoubleUnaryOpVertex(this, a -> a.acos(), a -> a.acos());
    }

    public DoubleVertex atan() {
        return new DoubleUnaryOpVertex(this, a -> a.atan(), a -> a.atan());
    }

    public DoubleVertex atan2(DoubleVertex that) {
        return new DoubleBinaryOpVertex(this, that, (a, b) -> a.atan2(b), (a, b) -> {
            DoubleTensor denominator = ((b.getValue().pow(2)).timesInPlace((a.getValue().pow(2))));

            PartialDerivatives thisInfA = a.getPartialDerivatives().multiplyBy(b.getValue().div(denominator));
            PartialDerivatives thisInfB = b.getPartialDerivatives().multiplyBy((a.getValue().div(denominator)).unaryMinusInPlace());
            PartialDerivatives newInf = thisInfA.add(thisInfB);
            return new DualNumber(a.getValue().atan2(b.getValue()), newInf);
        });
    }

    public DoubleVertex sum() {
        return new DoubleUnaryOpVertex(Tensor.SCALAR_SHAPE, this, a -> DoubleTensor.scalar(a.sum()), a -> a.sum());
    }

    public DoubleVertex reshape(int... proposedShape) {
        return new DoubleUnaryOpVertex(this, a -> a.reshape(proposedShape), a -> a.reshape(proposedShape));
    }

    public DoubleVertex lambda(int[] outputShape, Function<DoubleTensor, DoubleTensor> op, Function<Map<IVertex, DualNumber>, DualNumber> dualNumberCalculation) {
        return new DoubleUnaryOpLambda<>(outputShape, this, op, dualNumberCalculation);
    }
    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)

    @Override
    public DoubleVertex times(DoubleVertex that) {
        return multiply(that);
    }

    @Override
    public DoubleVertex div(DoubleVertex that) {
        return divideBy(that);
    }

    @Override
    public DoubleVertex times(double that) {
        return multiply(that);
    }

    @Override
    public DoubleVertex div(double that) {
        return divideBy(that);
    }

    @Override
    public DoubleVertex unaryMinus() {
        return multiply(-1.0);
    }


    public <T extends Tensor> BooleanVertex equalTo(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.elementwiseEquals(b));
    }

    public <T extends Tensor> BooleanVertex notEqualTo(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.elementwiseEquals(b).not());
    }

    public <T extends NumberTensor> BooleanVertex greaterThan(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.toDouble().greaterThan(b.toDouble()));
    }

    public <T extends NumberTensor> BooleanVertex greaterThanOrEqualTo(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.toDouble().greaterThanOrEqual(b.toDouble()));
    }

    public <T extends NumberTensor> BooleanVertex lessThan(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.toDouble().lessThan(b.toDouble()));
    }

    public <T extends NumberTensor> BooleanVertex lessThanOrEqualTo(Vertex<T> rhs) {
        return new BooleanBinaryOpVertex<>(this, rhs, (a, b) -> a.toDouble().lessThanOrEqual(b.toDouble()));
    }

    public void setValue(double value) {
        super.setValue(DoubleTensor.scalar(value));
    }

    public void setValue(double[] values) {
        super.setValue(DoubleTensor.create(values));
    }

    public void setAndCascade(double value) {
        super.setAndCascade(DoubleTensor.scalar(value));
    }

    public void setAndCascade(double[] values) {
        super.setAndCascade(DoubleTensor.create(values));
    }

    public void observe(double value) {
        this.observe(DoubleTensor.scalar(value));
    }

    public void observe(double[] values) {
        this.observe(DoubleTensor.create(values));
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
