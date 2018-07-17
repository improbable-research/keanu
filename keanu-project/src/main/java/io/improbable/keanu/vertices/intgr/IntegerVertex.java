package io.improbable.keanu.vertices.intgr;

import java.util.function.Function;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Observable;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerBinaryOpVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpLambda;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpVertex;
import io.improbable.keanu.vertices.update.ValueUpdater;

public abstract class IntegerVertex extends Vertex<IntegerTensor> implements IntegerOperators<IntegerVertex> {

    public IntegerVertex(ValueUpdater<IntegerTensor> valueUpdater, Observable<IntegerTensor> observation) {
        super(valueUpdater, observation);
    }

    @Override
    public IntegerVertex minus(IntegerVertex that) {
        return new IntegerBinaryOpVertex(this, that, (a,b) -> a.minus(b));
    }

    @Override
    public IntegerVertex plus(IntegerVertex that) {
        return new IntegerBinaryOpVertex(this, that, (a,b) -> a.plus(b));
    }

    public IntegerVertex multiply(IntegerVertex that) {
        return new IntegerBinaryOpVertex(this, that, (a,b) -> a.times(b));
    }

    public IntegerVertex divideBy(IntegerVertex that) {
        return new IntegerBinaryOpVertex(this, that, (a, b) -> a.div(b));
    }

    public IntegerVertex minus(Vertex<IntegerTensor> that) {
        return minus(new CastIntegerVertex(that));
    }

    public IntegerVertex plus(Vertex<IntegerTensor> that) {
        return plus(new CastIntegerVertex(that));
    }

    public IntegerVertex multiply(Vertex<IntegerTensor> that) {
        return multiply(new CastIntegerVertex(that));
    }

    public IntegerVertex divideBy(Vertex<IntegerTensor> that) {
        return divideBy(new CastIntegerVertex(that));
    }

    @Override
    public IntegerVertex pow(IntegerVertex that) {
        return new IntegerBinaryOpVertex(this, that, (a,b) -> a.pow(b));
    }

    public IntegerVertex minus(int value) {
        return minus(new ConstantIntegerVertex(value));
    }

    public IntegerVertex plus(int value) {
        return plus(new ConstantIntegerVertex(value));
    }

    public IntegerVertex multiply(int factor) {
        return multiply(new ConstantIntegerVertex(factor));
    }

    public IntegerVertex divideBy(int divisor) {
        return divideBy(new ConstantIntegerVertex(divisor));
    }

    public IntegerVertex pow(int exponent) {
        return pow(new ConstantIntegerVertex(exponent));
    }

    public IntegerVertex abs() {
        return new IntegerUnaryOpVertex(this, a -> a.abs());
    }

    public IntegerVertex sum() {
        return new IntegerUnaryOpVertex(this, a -> IntegerTensor.scalar(a.sum()));
    }

    public IntegerVertex lambda(int[] shape, Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(shape, this, op);
    }

    public IntegerVertex lambda(Function<IntegerTensor, IntegerTensor> op) {
        return lambda(this.getShape(), op);
    }

    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the Operators interface)
    @Override
    public IntegerVertex times(IntegerVertex that) {
        return multiply(that);
    }

    @Override
    public IntegerVertex div(IntegerVertex that) {
        return divideBy(that);
    }

    public IntegerVertex times(int that) {
        return multiply(that);
    }

    public IntegerVertex div(int that) {
        return divideBy(that);
    }

    @Override
    public IntegerVertex unaryMinus() {
        return multiply(-1);
    }

    public void setValue(int value) {
        super.setValue(IntegerTensor.create(value, getShape()));
    }

    public void setValue(int[] values) {
        super.setValue(IntegerTensor.create(values, getShape()));
    }

    public void setAndCascade(int value) {
        super.setAndCascade(IntegerTensor.create(value, getShape()));
    }

    public void setAndCascade(int[] values) {
        super.setAndCascade(IntegerTensor.create(values, getShape()));
    }

    public int getValue(int... index) {
        return getValue().getValue(index);
    }

}
