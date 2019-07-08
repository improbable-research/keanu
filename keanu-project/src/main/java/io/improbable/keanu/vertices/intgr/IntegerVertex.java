package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.FixedPointTensorVertex;
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

import java.util.List;
import java.util.function.Function;

public abstract class IntegerVertex extends Vertex<IntegerTensor> implements IntegerOperators<IntegerVertex>, FixedPointTensorVertex<Integer, IntegerVertex> {

    public IntegerVertex(long[] shape) {
        super(shape);
    }

    //////////////////////////
    ////  Vertex helpers
    //////////////////////////

    public static IntegerVertex concat(int dimension, IntegerVertex... toConcat) {
        return new IntegerConcatenationVertex(dimension, toConcat);
    }

    public static IntegerVertex min(IntegerVertex a, IntegerVertex b) {
        return new IntegerMinVertex(a, b);
    }

    public static IntegerVertex max(IntegerVertex a, IntegerVertex b) {
        return new IntegerMaxVertex(a, b);
    }

    @Override
    public void saveValue(NetworkSaver netSaver) {
        netSaver.saveValue(this);
    }

    @Override
    public void loadValue(NetworkLoader loader) {
        loader.loadValue(this);
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

    //////////////////////////
    ////  Tensor Operations
    //////////////////////////

    public void observe(int value) {
        super.observe(IntegerTensor.scalar(value));
    }

    public void observe(int[] values) {
        super.observe(IntegerTensor.create(values));
    }

    public int getValue(long... index) {
        return getValue().getValue(index);
    }

    @Override
    public IntegerVertex take(long... index) {
        return new IntegerTakeVertex(this, index);
    }

    @Override
    public List<IntegerVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    @Override
    public IntegerVertex diag() {
        return null;
    }

    @Override
    public IntegerVertex get(BooleanVertex booleanIndex) {
        return null;
    }

    @Override
    public IntegerVertex slice(int dimension, long index) {
        return new IntegerSliceVertex(this, dimension, index);
    }

    @Override
    public IntegerVertex slice(Slicer slicer) {
        return null;
    }

    @Override
    public IntegerVertex reshape(long... proposedShape) {
        return new IntegerReshapeVertex(this, proposedShape);
    }

    @Override
    public IntegerVertex permute(int... rearrange) {
        return null;
    }

    @Override
    public IntegerVertex broadcast(long... toShape) {
        return null;
    }

    @Override
    public BooleanVertex elementwiseEquals(IntegerVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    //////////////////////////
    ////  Number Tensor Operations
    //////////////////////////

    @Override
    public BooleanVertex elementwiseEquals(Integer value) {
        return elementwiseEquals(new ConstantIntegerVertex(value));
    }

    public BooleanVertex notEqualTo(IntegerVertex rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    @Override
    public IntegerVertex minus(IntegerVertex that) {
        return new IntegerDifferenceVertex(this, that);
    }

    @Override
    public IntegerVertex minus(int value) {
        return new IntegerDifferenceVertex(this, new ConstantIntegerVertex(value));
    }

    @Override
    public IntegerVertex minus(Integer value) {
        return null;
    }

    @Override
    public IntegerVertex reverseMinus(IntegerVertex value) {
        return null;
    }

    @Override
    public IntegerVertex reverseMinus(Integer value) {
        return null;
    }

    @Override
    public IntegerVertex reverseMinus(int that) {
        return new ConstantIntegerVertex(that).minus(this);
    }

    @Override
    public IntegerVertex unaryMinus() {
        return multiply(-1);
    }

    @Override
    public IntegerVertex plus(IntegerVertex that) {
        return new IntegerAdditionVertex(this, that);
    }

    @Override
    public IntegerVertex plus(int value) {
        return new IntegerAdditionVertex(this, new ConstantIntegerVertex(value));
    }

    @Override
    public IntegerVertex plus(Integer value) {
        return new IntegerAdditionVertex(this, new ConstantIntegerVertex(value));
    }

    public IntegerVertex multiply(IntegerVertex that) {
        return new IntegerMultiplicationVertex(this, that);
    }

    public IntegerVertex multiply(int factor) {
        return new IntegerMultiplicationVertex(this, new ConstantIntegerVertex(factor));
    }

    @Override
    public IntegerVertex times(IntegerVertex that) {
        return multiply(that);
    }

    @Override
    public IntegerVertex times(Integer value) {
        return multiply(value);
    }

    @Override
    public IntegerVertex times(int that) {
        return multiply(that);
    }

    public IntegerVertex divideBy(int divisor) {
        return new IntegerDivisionVertex(this, new ConstantIntegerVertex(divisor));
    }

    public IntegerVertex divideBy(IntegerVertex that) {
        return new IntegerDivisionVertex(this, that);
    }

    @Override
    public IntegerVertex div(IntegerVertex that) {
        return divideBy(that);
    }

    @Override
    public IntegerVertex div(Integer value) {
        return divideBy(value);
    }

    @Override
    public IntegerVertex div(int that) {
        return divideBy(that);
    }

    @Override
    public IntegerVertex reverseDiv(Integer value) {
        return null;
    }

    @Override
    public IntegerVertex reverseDiv(IntegerVertex value) {
        return null;
    }

    @Override
    public IntegerVertex reverseDiv(int that) {
        return (new ConstantIntegerVertex(that)).div(this);
    }

    @Override
    public IntegerVertex pow(IntegerVertex exponent) {
        return new IntegerPowerVertex(this, exponent);
    }

    @Override
    public IntegerVertex pow(Integer exponent) {
        return null;
    }

    @Override
    public IntegerVertex pow(int exponent) {
        return new IntegerPowerVertex(this, new ConstantIntegerVertex(exponent));
    }


    @Override
    public IntegerVertex average() {
        return null;
    }

    @Override
    public IntegerVertex standardDeviation() {
        return null;
    }

    @Override
    public IntegerVertex argMax(int axis) {
        return null;
    }

    @Override
    public IntegerVertex argMax() {
        return null;
    }

    @Override
    public IntegerVertex argMin(int axis) {
        return null;
    }

    @Override
    public IntegerVertex argMin() {
        return null;
    }

    @Override
    public IntegerVertex setWithMask(IntegerVertex mask, Integer value) {
        return null;
    }

    @Override
    public IntegerVertex apply(Function<Integer, Integer> function) {
        return null;
    }

    @Override
    public IntegerVertex safeLogTimes(IntegerVertex y) {
        return null;
    }

    @Override
    public BooleanVertex equalsWithinEpsilon(IntegerVertex other, Integer epsilon) {
        return null;
    }

    @Override
    public IntegerVertex abs() {
        return new IntegerAbsVertex(this);
    }

    @Override
    public IntegerVertex sum() {
        return new IntegerSumVertex(this);
    }

    @Override
    public IntegerVertex sum(int... sumOverDimensions) {
        return new IntegerSumVertex(this, sumOverDimensions);
    }

    @Override
    public IntegerVertex cumSum(int requestedDimension) {
        return null;
    }

    @Override
    public IntegerVertex product() {
        return null;
    }

    @Override
    public IntegerVertex product(int... overDimensions) {
        return null;
    }

    @Override
    public IntegerVertex cumProd(int requestedDimension) {
        return null;
    }

    @Override
    public IntegerVertex max() {
        return null;
    }

    @Override
    public IntegerVertex max(IntegerVertex max) {
        return null;
    }

    @Override
    public IntegerVertex min() {
        return null;
    }

    @Override
    public IntegerVertex min(IntegerVertex min) {
        return null;
    }

    @Override
    public IntegerVertex clamp(IntegerVertex min, IntegerVertex max) {
        return null;
    }

    @Override
    public IntegerVertex matrixMultiply(IntegerVertex that) {
        return null;
    }

    @Override
    public IntegerVertex tensorMultiply(IntegerVertex value, int[] dimLeft, int[] dimsRight) {
        return null;
    }

    public IntegerVertex lambda(long[] shape, Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(shape, this, op);
    }

    public IntegerVertex lambda(Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(this, op);
    }

    @Override
    public BooleanVertex toBoolean() {
        return null;
    }

    @Override
    public DoubleVertex toDouble() {
        return new CastToDoubleVertex(this);
    }

    @Override
    public IntegerVertex toInteger() {
        return this;
    }

    @Override
    public BooleanVertex greaterThan(IntegerVertex rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    @Override
    public BooleanVertex greaterThanOrEqual(IntegerVertex rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    @Override
    public BooleanVertex lessThan(Integer value) {
        return null;
    }

    @Override
    public BooleanVertex lessThanOrEqual(Integer value) {
        return null;
    }

    @Override
    public BooleanVertex greaterThan(Integer value) {
        return null;
    }

    @Override
    public BooleanVertex greaterThanOrEqual(Integer value) {
        return null;
    }

    @Override
    public BooleanVertex lessThan(IntegerVertex rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    @Override
    public BooleanVertex lessThanOrEqual(IntegerVertex rhs) {
        return new LessThanOrEqualVertex<>(this, rhs);
    }

    @Override
    public IntegerVertex greaterThanMask(IntegerVertex greaterThanThis) {
        return null;
    }

    @Override
    public IntegerVertex greaterThanOrEqualToMask(IntegerVertex greaterThanOrEqualThis) {
        return null;
    }

    @Override
    public IntegerVertex lessThanMask(IntegerVertex lessThanThis) {
        return null;
    }

    @Override
    public IntegerVertex lessThanOrEqualToMask(IntegerVertex lessThanOrEqualThis) {
        return null;
    }

    //////////////////////////
    ////  Fixed Point Tensor Operations
    //////////////////////////

    @Override
    public IntegerVertex mod(Integer that) {
        return null;
    }

    @Override
    public IntegerVertex mod(IntegerVertex that) {
        return null;
    }
}
