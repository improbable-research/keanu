package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerBinaryOpVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerGetBooleanIndexVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMinVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerPowerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerSetWithMaskVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerTensorMultiplyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.multiple.IntegerConcatenationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerAbsVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerApplyVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerBroadcastVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerPermuteVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerProductVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerReshapeVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSliceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerSumVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerTakeVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpLambda;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpVertex;
import io.improbable.keanu.vertices.number.FixedPointTensorVertex;

import java.util.List;
import java.util.function.Function;

import static io.improbable.keanu.tensor.TensorShapeValidation.getMatrixMultiplicationResultingShape;

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

    //////////////////////////
    ////  Tensor Operations
    //////////////////////////

    static IntegerVertex concat(int dimension, IntegerVertex... toConcat) {
        return new IntegerConcatenationVertex(dimension, toConcat);
    }

    @Override
    default IntegerVertex take(long... index) {
        return new IntegerTakeVertex(this, index);
    }

    @Override
    default List<IntegerVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    class IntegerDiagVertex extends IntegerUnaryOpVertex {
        @ExportVertexToPythonBindings
        public IntegerDiagVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex) {
            super(inputVertex.getShape(), inputVertex);
        }

        @Override
        protected IntegerTensor op(IntegerTensor value) {
            return value.diag();
        }
    }

    @Override
    default IntegerVertex diag() {
        return new IntegerDiagVertex(this);
    }

    @Override
    default IntegerVertex get(BooleanVertex booleanIndex) {
        return new IntegerGetBooleanIndexVertex(this, booleanIndex);
    }

    @Override
    default IntegerVertex slice(int dimension, long index) {
        return new IntegerSliceVertex(this, dimension, index);
    }

    @Override
    default IntegerVertex slice(Slicer slicer) {
        return null;
    }

    @Override
    default IntegerVertex reshape(long... proposedShape) {
        return new IntegerReshapeVertex(this, proposedShape);
    }

    @Override
    default IntegerVertex permute(int... rearrange) {
        return new IntegerPermuteVertex(this, rearrange);
    }

    @Override
    default IntegerVertex broadcast(long... toShape) {
        return new IntegerBroadcastVertex(this, toShape);
    }

    @Override
    default BooleanVertex elementwiseEquals(IntegerVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex elementwiseEquals(Integer value) {
        return elementwiseEquals(new ConstantIntegerVertex(value));
    }

    default BooleanVertex notEqualTo(IntegerVertex rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex notEqualTo(Integer value) {
        return notEqualTo(new ConstantIntegerVertex(value));
    }

    //////////////////////////
    ////  Number Tensor Operations
    //////////////////////////

    @Override
    default IntegerVertex minus(IntegerVertex that) {
        return new IntegerDifferenceVertex(this, that);
    }

    @Override
    default IntegerVertex minus(int value) {
        return new IntegerDifferenceVertex(this, new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex minus(Integer value) {
        return new IntegerDifferenceVertex(this, new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex reverseMinus(IntegerVertex value) {
        return value.minus(this);
    }

    @Override
    default IntegerVertex reverseMinus(Integer value) {
        return new ConstantIntegerVertex(value).minus(this);
    }

    @Override
    default IntegerVertex reverseMinus(int that) {
        return new ConstantIntegerVertex(that).minus(this);
    }

    @Override
    default IntegerVertex unaryMinus() {
        return multiply(-1);
    }

    @Override
    default IntegerVertex plus(IntegerVertex that) {
        return new IntegerAdditionVertex(this, that);
    }

    @Override
    default IntegerVertex plus(int value) {
        return new IntegerAdditionVertex(this, new ConstantIntegerVertex(value));
    }

    @Override
    default IntegerVertex plus(Integer value) {
        return new IntegerAdditionVertex(this, new ConstantIntegerVertex(value));
    }

    default IntegerVertex multiply(IntegerVertex that) {
        return new IntegerMultiplicationVertex(this, that);
    }

    default IntegerVertex multiply(int factor) {
        return new IntegerMultiplicationVertex(this, new ConstantIntegerVertex(factor));
    }

    @Override
    default IntegerVertex times(IntegerVertex that) {
        return multiply(that);
    }

    @Override
    default IntegerVertex times(Integer value) {
        return multiply(value);
    }

    @Override
    default IntegerVertex times(int that) {
        return multiply(that);
    }

    default IntegerVertex divideBy(int divisor) {
        return new IntegerDivisionVertex(this, new ConstantIntegerVertex(divisor));
    }

    default IntegerVertex divideBy(IntegerVertex that) {
        return new IntegerDivisionVertex(this, that);
    }

    @Override
    default IntegerVertex div(IntegerVertex that) {
        return divideBy(that);
    }

    @Override
    default IntegerVertex div(Integer value) {
        return divideBy(value);
    }

    @Override
    default IntegerVertex div(int that) {
        return divideBy(that);
    }

    @Override
    default IntegerVertex reverseDiv(Integer value) {
        return new ConstantIntegerVertex(value).div(this);
    }

    @Override
    default IntegerVertex reverseDiv(IntegerVertex value) {
        return value.div(this);
    }

    @Override
    default IntegerVertex reverseDiv(int that) {
        return (new ConstantIntegerVertex(that)).div(this);
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
    default IntegerVertex setWithMask(IntegerVertex mask, Integer value) {
        return new IntegerSetWithMaskVertex(this, mask, value);
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
    default IntegerVertex abs() {
        return new IntegerAbsVertex(this);
    }

    @Override
    default IntegerVertex sum() {
        return new IntegerSumVertex(this);
    }

    @Override
    default IntegerVertex sum(int... sumOverDimensions) {
        return new IntegerSumVertex(this, sumOverDimensions);
    }

    class IntegerCumSumVertex extends IntegerUnaryOpVertex {
        private final static String REQUESTED_DIMENSION = "requestedDimension";
        private final int requestedDimension;

        @ExportVertexToPythonBindings
        public IntegerCumSumVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex, int requestedDimension) {
            super(inputVertex.getShape(), inputVertex);
            this.requestedDimension = requestedDimension;
        }

        @Override
        protected IntegerTensor op(IntegerTensor value) {
            return value.cumSum(requestedDimension);
        }

        @SaveVertexParam(REQUESTED_DIMENSION)
        public int getRequestedDimension() {
            return requestedDimension;
        }
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

    class IntegerCumProdVertex extends IntegerUnaryOpVertex {
        private final static String REQUESTED_DIMENSION = "requestedDimension";
        private final int requestedDimension;

        @ExportVertexToPythonBindings
        public IntegerCumProdVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex, int requestedDimension) {
            super(inputVertex.getShape(), inputVertex);
            this.requestedDimension = requestedDimension;
        }

        @Override
        protected IntegerTensor op(IntegerTensor value) {
            return value.cumProd(requestedDimension);
        }

        @SaveVertexParam(REQUESTED_DIMENSION)
        public int getRequestedDimension() {
            return requestedDimension;
        }
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

    class IntegerMaxUnaryVertex extends IntegerUnaryOpVertex {
        @ExportVertexToPythonBindings
        public IntegerMaxUnaryVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex) {
            super(inputVertex.getShape(), inputVertex);
        }

        @Override
        protected IntegerTensor op(IntegerTensor value) {
            return value.max();
        }
    }

    @Override
    default IntegerVertex max() {
        return new IntegerMaxUnaryVertex(this);
    }

    @Override
    default IntegerVertex max(IntegerVertex that) {
        return max(this, that);
    }

    class IntegerMinUnaryVertex extends IntegerUnaryOpVertex {
        @ExportVertexToPythonBindings
        public IntegerMinUnaryVertex(@LoadVertexParam(INPUT_NAME) IntegerVertex inputVertex) {
            super(inputVertex.getShape(), inputVertex);
        }

        @Override
        protected IntegerTensor op(IntegerTensor value) {
            return value.min();
        }
    }

    @Override
    default IntegerVertex min() {
        return new IntegerMinUnaryVertex(this);
    }

    @Override
    default IntegerVertex min(IntegerVertex that) {
        return min(this, that);
    }

    @Override
    default IntegerVertex clamp(IntegerVertex min, IntegerVertex max) {
        return null;
    }

    class IntegerMatrixMultiplyVertex extends IntegerBinaryOpVertex {
        @ExportVertexToPythonBindings
        public IntegerMatrixMultiplyVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left, @LoadVertexParam(RIGHT_NAME) IntegerVertex right) {
            super(getMatrixMultiplicationResultingShape(left.getShape(), right.getShape()), left, right);
        }

        @Override
        protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
            return l.matrixMultiply(r);
        }
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

    @Override
    default BooleanVertex lessThan(Integer value) {
        return lessThan(new ConstantIntegerVertex(value));
    }

    @Override
    default BooleanVertex lessThanOrEqual(Integer value) {
        return lessThanOrEqual((new ConstantIntegerVertex(value)));
    }

    @Override
    default BooleanVertex greaterThan(Integer value) {
        return greaterThan((new ConstantIntegerVertex(value)));
    }

    @Override
    default BooleanVertex greaterThanOrEqual(Integer value) {
        return greaterThanOrEqual((new ConstantIntegerVertex(value)));
    }

    class IntegerGreaterThanMaskVertex extends IntegerBinaryOpVertex {

        @ExportVertexToPythonBindings
        public IntegerGreaterThanMaskVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left, @LoadVertexParam(RIGHT_NAME) IntegerVertex right) {
            super(left, right);
        }

        @Override
        protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
            return l.greaterThanMask(r);
        }
    }

    @Override
    default IntegerVertex greaterThanMask(IntegerVertex greaterThanThis) {
        return new IntegerGreaterThanMaskVertex(this, greaterThanThis);
    }

    class IntegerGreaterThanOrEqualMaskVertex extends IntegerBinaryOpVertex {

        @ExportVertexToPythonBindings
        public IntegerGreaterThanOrEqualMaskVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left, @LoadVertexParam(RIGHT_NAME) IntegerVertex right) {
            super(left, right);
        }

        @Override
        protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
            return l.greaterThanOrEqualToMask(r);
        }
    }

    @Override
    default IntegerVertex greaterThanOrEqualToMask(IntegerVertex greaterThanOrEqualThis) {
        return new IntegerGreaterThanOrEqualMaskVertex(this, greaterThanOrEqualThis);
    }

    class IntegerLessThanMaskVertex extends IntegerBinaryOpVertex {

        @ExportVertexToPythonBindings
        public IntegerLessThanMaskVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left, @LoadVertexParam(RIGHT_NAME) IntegerVertex right) {
            super(left, right);
        }

        @Override
        protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
            return l.lessThanMask(r);
        }
    }

    @Override
    default IntegerVertex lessThanMask(IntegerVertex lessThanThis) {
        return new IntegerLessThanMaskVertex(this, lessThanThis);
    }

    class IntegerLessThanOrEqualMaskVertex extends IntegerBinaryOpVertex {

        @ExportVertexToPythonBindings
        public IntegerLessThanOrEqualMaskVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left, @LoadVertexParam(RIGHT_NAME) IntegerVertex right) {
            super(left, right);
        }

        @Override
        protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
            return l.lessThanOrEqualToMask(r);
        }
    }

    @Override
    default IntegerVertex lessThanOrEqualToMask(IntegerVertex lessThanOrEqualThis) {
        return new IntegerLessThanOrEqualMaskVertex(this, lessThanOrEqualThis);
    }

    //////////////////////////
    ////  Fixed Point Tensor Operations
    //////////////////////////

    class IntegerModVertex extends IntegerBinaryOpVertex {

        @ExportVertexToPythonBindings
        public IntegerModVertex(@LoadVertexParam(LEFT_NAME) IntegerVertex left, @LoadVertexParam(RIGHT_NAME) IntegerVertex right) {
            super(left, right);
        }

        @Override
        protected IntegerTensor op(IntegerTensor l, IntegerTensor r) {
            return l.mod(r);
        }
    }

    @Override
    default IntegerVertex mod(Integer that) {
        return mod(new ConstantIntegerVertex(that));
    }

    @Override
    default IntegerVertex mod(IntegerVertex that) {
        return new IntegerModVertex(this, that);
    }

}
