package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastToDoubleVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastToIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMinVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerPowerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerAbsVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerReshapeVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSliceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSumVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerTakeVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpLambda;

import java.util.function.Function;

public abstract class IntegerVertex extends Vertex<IntegerTensor> implements IntegerOperators<IntegerVertex> {

    public IntegerVertex(long[] shape) {
        super(shape);
    }

    @Override
    public void saveValue(NetworkSaver netSaver) {
        netSaver.saveValue(this);
    }

    @Override
    public void loadValue(NetworkLoader loader) {
        loader.loadValue(this);
    }

    public static IntegerVertex concat(int dimension, IntegerVertex... toConcat) {
        return new IntegerConcatenationVertex(dimension, toConcat);
    }

    public static IntegerVertex min(IntegerVertex a, IntegerVertex b) {
        return new IntegerMinVertex(a, b);
    }

    public static IntegerVertex max(IntegerVertex a, IntegerVertex b) {
        return new IntegerMaxVertex(a, b);
    }

    public IntegerVertex minus(IntegerVertex that) {
        return new IntegerDifferenceVertex(this, that);
    }

    public IntegerVertex plus(IntegerVertex that) {
        return new IntegerAdditionVertex(this, that);
    }

    public IntegerVertex multiply(IntegerVertex that) {
        return new IntegerMultiplicationVertex(this, that);
    }

    public IntegerVertex divideBy(IntegerVertex that) {
        return new IntegerDivisionVertex(this, that);
    }

    public IntegerVertex minus(Vertex<IntegerTensor> that) {
        return new IntegerDifferenceVertex(this, new CastToIntegerVertex(that));
    }

    public IntegerVertex plus(Vertex<IntegerTensor> that) {
        return new IntegerAdditionVertex(this, new CastToIntegerVertex(that));
    }

    public IntegerVertex multiply(Vertex<IntegerTensor> that) {
        return new IntegerMultiplicationVertex(this, new CastToIntegerVertex(that));
    }

    public IntegerVertex divideBy(Vertex<IntegerTensor> that) {
        return new IntegerDivisionVertex(this, new CastToIntegerVertex(that));
    }

    @Override
    public IntegerVertex pow(IntegerVertex exponent) {
        return new IntegerPowerVertex(this, exponent);
    }

    public IntegerVertex minus(int value) {
        return new IntegerDifferenceVertex(this, new ConstantIntegerVertex(value));
    }

    public IntegerVertex plus(int value) {
        return new IntegerAdditionVertex(this, new ConstantIntegerVertex(value));
    }

    public IntegerVertex multiply(int factor) {
        return new IntegerMultiplicationVertex(this, new ConstantIntegerVertex(factor));
    }

    public IntegerVertex divideBy(int divisor) {
        return new IntegerDivisionVertex(this, new ConstantIntegerVertex(divisor));
    }

    public IntegerVertex pow(int exponent) {
        return new IntegerPowerVertex(this, new ConstantIntegerVertex(exponent));
    }

    public IntegerVertex abs() {
        return new IntegerAbsVertex(this);
    }

    public IntegerVertex sum() {
        return new IntegerSumVertex(this);
    }

    public IntegerVertex lambda(long[] shape, Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(shape, this, op);
    }

    public IntegerVertex lambda(Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(this, op);
    }

    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)
    public IntegerVertex times(IntegerVertex that) {
        return multiply(that);
    }

    public IntegerVertex reverseMinus(int that) {
        return new ConstantIntegerVertex(that).minus(this);
    }

    public IntegerVertex div(IntegerVertex that) {
        return divideBy(that);
    }

    public IntegerVertex reverseDiv(int that) {
        return (new ConstantIntegerVertex(that)).div(this);
    }

    public IntegerVertex times(int that) {
        return multiply(that);
    }

    public IntegerVertex div(int that) {
        return divideBy(that);
    }

    public IntegerVertex unaryMinus() {
        return multiply(-1);
    }

    public IntegerVertex take(long... index) {
        return new IntegerTakeVertex(this, index);
    }

    public IntegerVertex slice(int dimension, int index) {
        return new IntegerSliceVertex(this, dimension, index);
    }

    public IntegerVertex reshape(long... proposedShape) {
        return new IntegerReshapeVertex(this, proposedShape);
    }

    public BooleanVertex equalTo(IntegerVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    public DoubleVertex toDouble() { return new CastToDoubleVertex(this); }

    public <T extends Tensor> BooleanVertex notEqualTo(Vertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BooleanVertex greaterThan(Vertex<T> rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BooleanVertex greaterThanOrEqualTo(Vertex<T> rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BooleanVertex lessThan(Vertex<T> rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BooleanVertex lessThanOrEqualTo(Vertex<T> rhs) {
        return new LessThanOrEqualVertex<>(this, rhs);
    }

    public void setValue(int value) {
        super.setValue(IntegerTensor.scalar(value));
    }

    public void setValue(int[] values) {
        super.setValue(IntegerTensor.create(values));
    }

    public void setAndCascade(int value) {
        super.setAndCascade(IntegerTensor.scalar(value));
    }

    public void setAndCascade(int[] values) {
        super.setAndCascade(IntegerTensor.create(values));
    }

    public void observe(int value) {
        super.observe(IntegerTensor.scalar(value));
    }

    public void observe(int[] values) {
        super.observe(IntegerTensor.create(values));
    }

    public int getValue(int... index) {
        return getValue().getValue(index);
    }

}
