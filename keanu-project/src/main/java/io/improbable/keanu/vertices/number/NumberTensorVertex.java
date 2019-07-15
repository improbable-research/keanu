package io.improbable.keanu.vertices.number;

import io.improbable.keanu.BaseNumberTensor;
import io.improbable.keanu.kotlin.NumberOperators;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.CastNumberToBooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NumericalEqualsVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastNumberToDoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastNumberToIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.ArgMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.ArgMinVertex;
import io.improbable.keanu.vertices.number.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.number.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.number.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.number.operators.binary.GreaterThanMaskVertex;
import io.improbable.keanu.vertices.number.operators.binary.GreaterThanOrEqualToMaskVertex;
import io.improbable.keanu.vertices.number.operators.binary.LessThanMaskVertex;
import io.improbable.keanu.vertices.number.operators.binary.LessThanOrEqualToMaskVertex;
import io.improbable.keanu.vertices.number.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.number.operators.binary.MaxVertex;
import io.improbable.keanu.vertices.number.operators.binary.MinVertex;
import io.improbable.keanu.vertices.number.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.number.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.number.operators.binary.TensorMultiplicationVertex;
import io.improbable.keanu.vertices.number.operators.ternary.SetWithMaskVertex;
import io.improbable.keanu.vertices.number.operators.unary.AbsVertex;
import io.improbable.keanu.vertices.number.operators.unary.ApplyVertex;
import io.improbable.keanu.vertices.number.operators.unary.CumProdVertex;
import io.improbable.keanu.vertices.number.operators.unary.CumSumVertex;
import io.improbable.keanu.vertices.number.operators.unary.MaxUnaryVertex;
import io.improbable.keanu.vertices.number.operators.unary.MinUnaryVertex;
import io.improbable.keanu.vertices.number.operators.unary.ProductVertex;
import io.improbable.keanu.vertices.number.operators.unary.SumVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;

import java.util.function.Function;

public interface NumberTensorVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends TensorVertex<T, TENSOR, VERTEX>, BaseNumberTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX>, NumberOperators<VERTEX> {

    @Override
    default VERTEX minus(VERTEX that) {
        return wrap(new DifferenceVertex<>(this, that));
    }

    @Override
    default VERTEX reverseMinus(VERTEX that) {
        return wrap(new DifferenceVertex<>(that, this));
    }

    @Override
    default VERTEX plus(VERTEX that) {
        return wrap(new AdditionVertex<>(this, that));
    }

    @Override
    default VERTEX times(VERTEX that) {
        return wrap(new MultiplicationVertex<>(this, that));
    }

    @Override
    default VERTEX div(VERTEX that) {
        return wrap(new DivisionVertex<>(this, that));
    }

    @Override
    default VERTEX reverseDiv(VERTEX that) {
        return wrap(new DivisionVertex<>(that, this));
    }

    @Override
    default VERTEX abs() {
        return wrap(new AbsVertex<>(this));
    }

    @Override
    default VERTEX sum() {
        return wrap(new SumVertex<>(this));
    }

    @Override
    default VERTEX sum(int... sumOverDimensions) {
        return wrap(new SumVertex<>(this, sumOverDimensions));
    }

    @Override
    default BooleanVertex toBoolean() {
        return new CastNumberToBooleanVertex<>(this);
    }

    @Override
    default DoubleVertex toDouble() {
        return new CastNumberToDoubleVertex<>(this);
    }

    @Override
    default IntegerVertex toInteger() {
        return new CastNumberToIntegerVertex<>(this);
    }

    @Override
    default BooleanVertex equalsWithinEpsilon(VERTEX other, T epsilon) {
        return new NumericalEqualsVertex<>(this, other, epsilon);
    }

    @Override
    default BooleanVertex greaterThan(VERTEX rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex greaterThanOrEqual(VERTEX rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex lessThan(VERTEX rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex lessThanOrEqual(VERTEX rhs) {
        return new LessThanOrEqualVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex lessThan(T value) {
        return lessThan((VERTEX) ConstantVertex.scalar(value));
    }

    @Override
    default BooleanVertex lessThanOrEqual(T value) {
        return lessThanOrEqual((VERTEX) ConstantVertex.scalar(value));
    }

    @Override
    default BooleanVertex greaterThan(T value) {
        return greaterThan((VERTEX) ConstantVertex.scalar(value));
    }

    @Override
    default BooleanVertex greaterThanOrEqual(T value) {
        return greaterThanOrEqual((VERTEX) ConstantVertex.scalar(value));
    }

    @Override
    default VERTEX greaterThanMask(VERTEX rhs) {
        return wrap(new GreaterThanMaskVertex<>(this, rhs));
    }

    @Override
    default VERTEX greaterThanOrEqualToMask(VERTEX rhs) {
        return wrap(new GreaterThanOrEqualToMaskVertex<>(this, rhs));
    }

    @Override
    default VERTEX lessThanMask(VERTEX rhs) {
        return wrap(new LessThanMaskVertex<>(this, rhs));
    }

    @Override
    default VERTEX lessThanOrEqualToMask(VERTEX rhs) {
        return wrap(new LessThanOrEqualToMaskVertex<>(this, rhs));
    }

    @Override
    default VERTEX setWithMask(VERTEX mask, T value) {
        return wrap(new SetWithMaskVertex<>(this, mask, ConstantVertex.scalar(value)));
    }

    default VERTEX setWithMask(VERTEX mask, VERTEX value) {
        return wrap(new SetWithMaskVertex<>(this, mask, value));
    }

    @Override
    default VERTEX max(VERTEX that) {
        return wrap(new MaxVertex<>(this, that));
    }

    @Override
    default VERTEX min(VERTEX that) {
        return wrap(new MinVertex<>(this, that));
    }

    @Override
    default VERTEX max() {
        return wrap(new MaxUnaryVertex<>(this));
    }

    @Override
    default VERTEX min() {
        return wrap(new MinUnaryVertex<>(this));
    }


    @Override
    default VERTEX pow(VERTEX exponent) {
        return wrap(new PowerVertex<>(this, exponent));
    }

    @Override
    default VERTEX pow(T exponent) {
        return pow((VERTEX) ConstantVertex.scalar(exponent));
    }

    /**
     * Matrix product of two vertices
     *
     * @param that a double vertex representing a matrix or a vector to matrix multiply
     * @return a vertex that represents the matrix multiplication of two vertices.
     * - If both left and right operands are rank 1, they are promoted to a matrix by prepending a 1 to its dimensions.
     * After matrix multiplication, it is reshaped to be a scalar. This is essentially a dot product.
     * This returns a ReshapeVertex.
     * - If only one of the operands is rank 1 (and the other operand is rank 2), it is promoted to a matrix by prepending a 1 to its dimensions.
     * After matrix multiplication, the appended 1 is removed. This is essentially a matrix-vector product.
     * This returns a ReshapeVertex.
     * - Otherwise, they are multiplied like conventional matrices.
     * This returns a MatrixMultiplicationVertex.
     */
    @Override
    default VERTEX matrixMultiply(VERTEX that) {
        int leftRank = this.getRank();
        int rightRank = that.getRank();

        if (leftRank < 1 || rightRank < 1) {
            throw new IllegalArgumentException("Matrix multiply for rank 0 is not supported. Use times instead.");
        }

        TensorVertex<T, TENSOR, VERTEX> leftMatrix = leftRank == 1 ? this.reshape(1, this.getShape()[0]) : this;
        TensorVertex<T, TENSOR, VERTEX> rightMatrix = rightRank == 1 ? that.reshape(that.getShape()[0], 1) : that;

        VERTEX result = wrap(new MatrixMultiplicationVertex<>(leftMatrix, rightMatrix));

        if (leftRank == 1 && rightRank == 1) {
            return result.reshape(new long[0]);
        } else if (leftRank == 1 && rightRank == 2) {
            return result.reshape(result.getShape()[1]);
        } else if (leftRank == 2 && rightRank == 1) {
            return result.reshape(result.getShape()[0]);
        } else {
            return result;
        }
    }

    @Override
    default VERTEX tensorMultiply(VERTEX value, int[] dimLeft, int[] dimsRight) {
        return wrap(new TensorMultiplicationVertex<>(this, value, dimLeft, dimsRight));
    }

    @Override
    default VERTEX apply(Function<T, T> function) {
        return wrap(new ApplyVertex<>(this, function));
    }

    @Override
    default VERTEX cumSum(int requestedDimension) {
        return wrap(new CumSumVertex<>(this, requestedDimension));
    }

    @Override
    default VERTEX product() {
        return wrap(new ProductVertex<>(this));
    }

    @Override
    default VERTEX product(int... overDimensions) {
        return wrap(new ProductVertex<>(this, overDimensions));
    }

    @Override
    default VERTEX cumProd(int requestedDimension) {
        return wrap(new CumProdVertex<>(this, requestedDimension));
    }

    @Override
    default IntegerVertex argMax(int axis) {
        return new ArgMaxVertex<>(this, axis);
    }

    @Override
    default IntegerVertex argMax() {
        return new ArgMaxVertex<>(this);
    }

    @Override
    default IntegerVertex argMin(int axis) {
        return new ArgMinVertex<>(this, axis);
    }

    @Override
    default IntegerVertex argMin() {
        return new ArgMinVertex<>(this);
    }


}
