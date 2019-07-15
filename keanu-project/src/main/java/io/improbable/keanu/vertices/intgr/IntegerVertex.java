package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMatrixMultiplyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMinVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerModVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerPowerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerTensorMultiplyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerApplyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerCumProdVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerCumSumVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerProductVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpLambda;
import io.improbable.keanu.vertices.number.FixedPointTensorVertex;

import java.util.List;
import java.util.function.Function;

public interface IntegerVertex extends IntegerOperators<IntegerVertex>, FixedPointTensorVertex<Integer, IntegerTensor, IntegerVertex> {

    //////////////////////////
    ////  Vertex Helpers
    //////////////////////////

    default void setValue(int value) {
        setValue(IntegerTensor.scalar(value));
    }

    default void setValue(int[] values) {
        setValue(IntegerTensor.create(values));
    }

    default void setAndCascade(int value) {
        setAndCascade(IntegerTensor.scalar(value));
    }

    default void setAndCascade(int[] values) {
        setAndCascade(IntegerTensor.create(values));
    }

    default void observe(int value) {
        observe(IntegerTensor.scalar(value));
    }

    default void observe(int[] values) {
        observe(IntegerTensor.create(values));
    }

    default int getValue(long... index) {
        return getValue().getValue(index);
    }

    @Override
    default IntegerVertex wrap(NonProbabilisticVertex<IntegerTensor, IntegerVertex> vertex) {
        return new IntegerVertexWrapper(vertex);
    }

    @Override
    default Class<?> ofType() {
        return IntegerTensor.class;
    }

    //////////////////////////
    ////  Tensor Operations
    //////////////////////////

    static IntegerVertex concat(int dimension, IntegerVertex... toConcat) {
        return new IntegerConcatenationVertex(dimension, toConcat);
    }

    @Override
    default List<IntegerVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    @Override
    default IntegerVertex slice(Slicer slicer) {
        return null;
    }

    //////////////////////////
    ////  Number Tensor Operations
    //////////////////////////

    @Override
    default IntegerVertex minus(int value) {
        return minus(new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex minus(Integer value) {
        return minus(new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex reverseMinus(Integer value) {
        return reverseMinus(new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex reverseMinus(int that) {
        return reverseMinus(new ConstantIntegerVertex(that));
    }

    @Override
    default IntegerVertex unaryMinus() {
        return multiply(-1);
    }

    @Override
    default IntegerVertex plus(int value) {
        return plus(new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex plus(Integer value) {
        return plus(new ConstantIntegerVertex(value));
    }

    default IntegerVertex multiply(IntegerVertex factor) {
        return times(factor);
    }

    default IntegerVertex multiply(int factor) {
        return times(new ConstantIntegerVertex(factor));
    }

    default IntegerVertex multiply(Integer factor) {
        return times(new ConstantIntegerVertex(factor));
    }

    @Override
    default IntegerVertex times(Integer value) {
        return times(new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex times(int that) {
        return times(new ConstantIntegerVertex(that));
    }

    default IntegerVertex divideBy(Integer divisor) {
        return div(new ConstantIntegerVertex(divisor));
    }

    default IntegerVertex divideBy(int divisor) {
        return div(new ConstantIntegerVertex(divisor));
    }

    default IntegerVertex divideBy(IntegerVertex that) {
        return div(that);
    }

    @Override
    default IntegerVertex div(Integer value) {
        return div(new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex div(int that) {
        return div(new ConstantIntegerVertex(that));
    }

    @Override
    default IntegerVertex reverseDiv(Integer value) {
        return reverseDiv(new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex reverseDiv(int value) {
        return reverseDiv(new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex pow(IntegerVertex exponent) {
        return new IntegerPowerVertex(this, exponent);
    }

    @Override
    default IntegerVertex pow(Integer exponent) {
        return pow(new ConstantIntegerVertex(exponent));
    }

    @Override
    default IntegerVertex pow(int exponent) {
        return pow(new ConstantIntegerVertex(exponent));
    }

    @Override
    default IntegerVertex average() {
        return null;
    }

    @Override
    default IntegerVertex standardDeviation() {
        return null;
    }

    @Override
    default IntegerVertex argMax(int axis) {
        return null;
    }

    @Override
    default IntegerVertex argMax() {
        return null;
    }

    @Override
    default IntegerVertex argMin(int axis) {
        return null;
    }

    @Override
    default IntegerVertex argMin() {
        return null;
    }

    @Override
    default IntegerVertex apply(Function<Integer, Integer> function) {
        return new IntegerApplyVertex(this, function);
    }

    @Override
    default IntegerVertex safeLogTimes(IntegerVertex y) {
        return null;
    }

    @Override
    default IntegerVertex cumSum(int requestedDimension) {
        return new IntegerCumSumVertex(this, requestedDimension);
    }

    @Override
    default IntegerVertex product() {
        return new IntegerProductVertex(this);
    }

    @Override
    default IntegerVertex product(int... overDimensions) {
        return new IntegerProductVertex(this, overDimensions);
    }

    @Override
    default IntegerVertex cumProd(int requestedDimension) {
        return new IntegerCumProdVertex(this, requestedDimension);
    }

    static IntegerVertex min(IntegerVertex a, IntegerVertex b) {
        return new IntegerMinVertex(a, b);
    }

    static IntegerVertex max(IntegerVertex a, IntegerVertex b) {
        return new IntegerMaxVertex(a, b);
    }

    @Override
    default IntegerVertex clamp(IntegerVertex min, IntegerVertex max) {
        return null;
    }

    @Override
    default IntegerVertex matrixMultiply(IntegerVertex that) {
        return new IntegerMatrixMultiplyVertex(this, that);
    }

    @Override
    default IntegerVertex tensorMultiply(IntegerVertex value, int[] dimLeft, int[] dimsRight) {
        return new IntegerTensorMultiplyVertex(this, value, dimLeft, dimsRight);
    }

    default IntegerVertex lambda(long[] shape, Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda(shape, this, op);
    }

    default IntegerVertex lambda(Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda(this, op);
    }

    //////////////////////////
    ////  Fixed Point Tensor Operations
    //////////////////////////

    @Override
    default IntegerVertex mod(Integer that) {
        return mod(new ConstantIntegerVertex(that));
    }

    @Override
    default IntegerVertex mod(IntegerVertex that) {
        return new IntegerModVertex(this, that);
    }

}
