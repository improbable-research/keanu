package io.improbable.keanu.vertices.intgr;

import java.util.function.Function;

import com.google.common.primitives.Ints;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexSampler;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerPowerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerAbsVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerReshapeVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSliceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSumVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerTakeVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpLambda;

public abstract class IntegerVertex extends Vertex<IntegerTensor> implements IntegerOperators<IntegerVertex> {

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
        return new IntegerDifferenceVertex(this, new CastIntegerVertex(that));
    }

    public IntegerVertex plus(Vertex<IntegerTensor> that) {
        return new IntegerAdditionVertex(this, new CastIntegerVertex(that));
    }

    public IntegerVertex multiply(Vertex<IntegerTensor> that) {
        return new IntegerMultiplicationVertex(this, new CastIntegerVertex(that));
    }

    public IntegerVertex divideBy(Vertex<IntegerTensor> that) {
        return new IntegerDivisionVertex(this, new CastIntegerVertex(that));
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

    public IntegerVertex lambda(int[] shape, Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(shape, this, op);
    }

    public IntegerVertex lambda(Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(this, op);
    }

    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)
    public IntegerVertex times(IntegerVertex that) {
        return multiply(that);
    }

    public IntegerVertex div(IntegerVertex that) {
        return divideBy(that);
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

    public IntegerVertex take(int... index) {
        return new IntegerTakeVertex(this, index);
    }

    public IntegerVertex slice(int dimension, int index) {
        return new IntegerSliceVertex(this, dimension, index);
    }

    public IntegerVertex reshape(int... proposedShape) {
        return new IntegerReshapeVertex(this, proposedShape);
    }

    public BoolVertex equalTo(IntegerVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    public <T extends Tensor> BoolVertex notEqualTo(Vertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BoolVertex greaterThan(Vertex<T> rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BoolVertex greaterThanOrEqualTo(Vertex<T> rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BoolVertex lessThan(Vertex<T> rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BoolVertex lessThanOrEqualTo(Vertex<T> rhs) {
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

    public IntegerTensor sampleScalarValuesAsTensor(int[] shape, KeanuRandom random) {
        final int length = Math.toIntExact(TensorShape.getLength(shape));
        final int[] samples = Ints.toArray(
            VertexSampler.sampleManyScalarsFromTensorVertex(this, length, random));
        return IntegerTensor.create(samples, shape);
    }

    public IntegerTensor sampleScalarValuesAsTensor(int[] shape) {
        return sampleScalarValuesAsTensor(shape, KeanuRandom.getDefaultRandom());
    }

}
