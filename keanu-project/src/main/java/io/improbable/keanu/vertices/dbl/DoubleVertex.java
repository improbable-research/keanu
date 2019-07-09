package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.IVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.NumericalEqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.EqualsVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.GreaterThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.NotEqualsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleGreaterThanMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleGreaterThanOrEqualToMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleLessThanMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DoubleLessThanOrEqualToMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MatrixMultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MaxVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.ternary.DoubleSetWithMaskVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ArcTanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.CosVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogGammaVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.LogVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixDeterminantVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.MatrixInverseVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.PermuteVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.ReshapeVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SliceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SumVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TakeVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TanVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastToIntegerVertex;
import io.improbable.keanu.vertices.number.FloatingPointTensorVertex;

import java.util.List;
import java.util.Map;
import java.util.function.Function;

public interface DoubleVertex extends DoubleOperators<DoubleVertex>, FloatingPointTensorVertex<Double, DoubleTensor, DoubleVertex> {

    //////////////////////////
    ////  Vertex helpers
    //////////////////////////

    default void setValue(double value) {
        setValue(DoubleTensor.scalar(value));
    }

    default void setValue(double[] values) {
        setValue(DoubleTensor.create(values));
    }

    default void setAndCascade(double value) {
        setAndCascade(DoubleTensor.scalar(value));
    }

    default void setAndCascade(double[] values) {
        setAndCascade(DoubleTensor.create(values));
    }

    default void observe(double value) {
        observe(DoubleTensor.scalar(value));
    }

    default void observe(double[] values) {
        observe(DoubleTensor.create(values));
    }

    default double getValue(long... index) {
        return getValue().getValue(index);
    }

    @Override
    default void loadValue(NetworkLoader loader) {
        loader.loadValue(this);
    }

    @Override
    default void saveValue(NetworkSaver netSaver) {
        netSaver.saveValue(this);
    }

    //////////////////////////
    ////  Tensor Operations
    //////////////////////////

    /**
     * @param dimension dimension to concat along. Negative dimension indexing is not supported.
     * @param toConcat  array of things to concat. Must match in all dimensions except for the provided
     *                  dimension
     * @return a vertex that represents the concatenation of the toConcat
     */
    static ConcatenationVertex concat(int dimension, DoubleVertex... toConcat) {
        return new ConcatenationVertex(dimension, toConcat);
    }

    @Override
    default ReshapeVertex reshape(long... proposedShape) {
        return new ReshapeVertex(this, proposedShape);
    }

    @Override
    default PermuteVertex permute(int... rearrange) {
        return new PermuteVertex(this, rearrange);
    }

    @Override
    default DoubleVertex broadcast(long... toShape) {
        return null;
    }

    @Override
    default PermuteVertex transpose() {
        return new PermuteVertex(this, 1, 0);
    }

    @Override
    default TakeVertex take(long... index) {
        return new TakeVertex(this, index);
    }

    @Override
    default List<DoubleVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    @Override
    default DoubleVertex diag() {
        return null;
    }

    @Override
    default DoubleVertex get(BooleanVertex booleanIndex) {
        return null;
    }

    @Override
    default SliceVertex slice(int dimension, long index) {
        return new SliceVertex(this, dimension, index);
    }

    @Override
    default DoubleVertex slice(Slicer slicer) {
        return null;
    }

    @Override
    default BooleanVertex elementwiseEquals(DoubleVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    @Override
    default BooleanVertex elementwiseEquals(Double value) {
        return new EqualsVertex<>(this, new ConstantDoubleVertex(value));
    }

    default <T extends Tensor> BooleanVertex notEqualTo(IVertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    //////////////////////////
    ////  Number Tensor Operations
    //////////////////////////

    @Override
    default SumVertex sum() {
        return new SumVertex(this);
    }

    @Override
    default SumVertex sum(int... sumOverDimensions) {
        return new SumVertex(this, sumOverDimensions);
    }

    @Override
    default DoubleVertex cumSum(int requestedDimension) {
        return null;
    }

    @Override
    default DoubleVertex product() {
        return null;
    }

    @Override
    default DoubleVertex product(int... overDimensions) {
        return null;
    }

    @Override
    default DoubleVertex cumProd(int requestedDimension) {
        return null;
    }

    static MinVertex min(DoubleVertex a, DoubleVertex b) {
        return new MinVertex(a, b);
    }

    static MaxVertex max(DoubleVertex a, DoubleVertex b) {
        return new MaxVertex(a, b);
    }

    @Override
    default DoubleVertex max() {
        return null;
    }

    @Override
    default DoubleVertex max(DoubleVertex that) {
        return max(this, that);
    }

    @Override
    default DoubleVertex min() {
        return null;
    }

    @Override
    default DoubleVertex min(DoubleVertex that) {
        return min(this, that);
    }

    @Override
    default DoubleVertex clamp(DoubleVertex min, DoubleVertex max) {
        return null;
    }

    @Override
    default DifferenceVertex minus(double that) {
        return minus(new ConstantDoubleVertex(that));
    }

    @Override
    default DifferenceVertex minus(DoubleVertex that) {
        return new DifferenceVertex(this, that);
    }

    @Override
    default DoubleVertex minus(Double value) {
        return null;
    }

    @Override
    default MultiplicationVertex unaryMinus() {
        return multiply(-1.0);
    }

    @Override
    default DoubleVertex reverseMinus(DoubleVertex value) {
        return null;
    }

    @Override
    default DoubleVertex reverseMinus(Double value) {
        return null;
    }

    @Override
    default DifferenceVertex reverseMinus(double that) {
        return new ConstantDoubleVertex(that).minus(this);
    }

    @Override
    default DoubleVertex plus(double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    @Override
    default AdditionVertex plus(Double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    @Override
    default AdditionVertex plus(DoubleVertex that) {
        return new AdditionVertex(this, that);
    }

    default MultiplicationVertex multiply(double that) {
        return multiply(new ConstantDoubleVertex(that));
    }

    default MultiplicationVertex multiply(DoubleVertex that) {
        return new MultiplicationVertex(this, that);
    }

    @Override
    default MultiplicationVertex times(DoubleVertex that) {
        return multiply(that);
    }

    @Override
    default MultiplicationVertex times(Double value) {
        return multiply(value);
    }

    @Override
    default MultiplicationVertex times(double that) {
        return multiply(that);
    }

    @Override
    default DivisionVertex div(Double value) {
        return divideBy(value);
    }

    default DivisionVertex divideBy(double that) {
        return divideBy(new ConstantDoubleVertex(that));
    }

    default DivisionVertex divideBy(DoubleVertex that) {
        return new DivisionVertex(this, that);
    }

    @Override
    default DivisionVertex div(DoubleVertex that) {
        return divideBy(that);
    }

    @Override
    default DivisionVertex div(double that) {
        return divideBy(that);
    }

    @Override
    default DivisionVertex reverseDiv(Double value) {
        return null;
    }

    @Override
    default DivisionVertex reverseDiv(DoubleVertex value) {
        return null;
    }

    @Override
    default DivisionVertex reverseDiv(double that) {
        return new ConstantDoubleVertex(that).div(this);
    }

    @Override
    default PowerVertex pow(double that) {
        return pow(new ConstantDoubleVertex(that));
    }

    @Override
    default PowerVertex pow(DoubleVertex exponent) {
        return new PowerVertex(this, exponent);
    }

    @Override
    default PowerVertex pow(Double exponent) {
        return null;
    }

    @Override
    default DoubleVertex average() {
        return null;
    }

    @Override
    default DoubleVertex standardDeviation() {
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
    default AbsVertex abs() {
        return new AbsVertex(this);
    }

    @Override
    default GreaterThanVertex greaterThan(DoubleVertex rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    @Override
    default GreaterThanOrEqualVertex greaterThanOrEqual(DoubleVertex rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    @Override
    default GreaterThanVertex greaterThan(Double value) {
        return new GreaterThanVertex<>(this, new ConstantDoubleVertex(value));
    }

    @Override
    default GreaterThanOrEqualVertex greaterThanOrEqual(Double value) {
        return new GreaterThanOrEqualVertex<>(this, new ConstantDoubleVertex(value));
    }

    @Override
    default LessThanVertex lessThan(Double value) {
        return new LessThanVertex<>(this, new ConstantDoubleVertex(value));
    }

    @Override
    default LessThanOrEqualVertex lessThanOrEqual(Double value) {
        return new LessThanOrEqualVertex<>(this, new ConstantDoubleVertex(value));
    }

    @Override
    default LessThanVertex lessThan(DoubleVertex rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    @Override
    default LessThanOrEqualVertex lessThanOrEqual(DoubleVertex rhs) {
        return new LessThanOrEqualVertex<>(this, rhs);
    }

    @Override
    default DoubleGreaterThanMaskVertex greaterThanMask(DoubleVertex rhs) {
        return new DoubleGreaterThanMaskVertex(this, rhs);
    }

    default DoubleGreaterThanMaskVertex greaterThanMask(Double rhs) {
        return greaterThanMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    default DoubleGreaterThanOrEqualToMaskVertex greaterThanOrEqualToMask(DoubleVertex rhs) {
        return new DoubleGreaterThanOrEqualToMaskVertex(this, rhs);
    }

    default DoubleGreaterThanOrEqualToMaskVertex greaterThanOrEqualToMask(Double rhs) {
        return greaterThanOrEqualToMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    default DoubleLessThanMaskVertex lessThanMask(DoubleVertex rhs) {
        return new DoubleLessThanMaskVertex(this, rhs);
    }

    default DoubleLessThanMaskVertex lessThanMask(Double rhs) {
        return lessThanMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    default DoubleLessThanOrEqualToMaskVertex lessThanOrEqualToMask(DoubleVertex rhs) {
        return new DoubleLessThanOrEqualToMaskVertex(this, rhs);
    }

    default DoubleLessThanOrEqualToMaskVertex lessThanOrEqualToMask(Double rhs) {
        return lessThanOrEqualToMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    default DoubleSetWithMaskVertex setWithMask(DoubleVertex mask, Double value) {
        return setWithMask(mask, new ConstantDoubleVertex(value));
    }

    @Override
    default DoubleVertex apply(Function<Double, Double> function) {
        return null;
    }

    @Override
    default DoubleVertex safeLogTimes(DoubleVertex y) {
        return null;
    }

    @Override
    default NumericalEqualsVertex equalsWithinEpsilon(DoubleVertex other, Double epsilon) {
        return new NumericalEqualsVertex<>(this, other, epsilon);
    }

    default DoubleSetWithMaskVertex setWithMask(DoubleVertex mask, DoubleVertex value) {
        return new DoubleSetWithMaskVertex(this, mask, value);
    }

    //////////////////////////
    ////  Floating Point Tensor Operations
    //////////////////////////

    @Override
    default FloorVertex floor() {
        return new FloorVertex(this);
    }

    @Override
    default CeilVertex ceil() {
        return new CeilVertex(this);
    }

    @Override
    default RoundVertex round() {
        return new RoundVertex(this);
    }

    @Override
    default ExpVertex exp() {
        return new ExpVertex(this);
    }

    @Override
    default DoubleVertex logAddExp2(DoubleVertex that) {
        return null;
    }

    @Override
    default DoubleVertex logAddExp(DoubleVertex that) {
        return null;
    }

    @Override
    default DoubleVertex log1p() {
        return null;
    }

    @Override
    default DoubleVertex log2() {
        return null;
    }

    @Override
    default DoubleVertex log10() {
        return null;
    }

    @Override
    default DoubleVertex exp2() {
        return null;
    }

    @Override
    default DoubleVertex expM1() {
        return null;
    }

    @Override
    default DoubleVertex replaceNaN(Double value) {
        return null;
    }

    @Override
    default BooleanVertex notNaN() {
        return null;
    }

    @Override
    default BooleanVertex isNaN() {
        return null;
    }

    @Override
    default BooleanVertex isFinite() {
        return null;
    }

    @Override
    default BooleanVertex isInfinite() {
        return null;
    }

    @Override
    default BooleanVertex isNegativeInfinity() {
        return null;
    }

    @Override
    default BooleanVertex isPositiveInfinity() {
        return null;
    }

    @Override
    default IntegerVertex nanArgMax(int axis) {
        return null;
    }

    @Override
    default IntegerVertex nanArgMax() {
        return null;
    }

    @Override
    default IntegerVertex nanArgMin(int axis) {
        return null;
    }

    @Override
    default IntegerVertex nanArgMin() {
        return null;
    }

    @Override
    default DoubleVertex reciprocal() {
        return null;
    }

    @Override
    default PowerVertex sqrt() {
        return new PowerVertex(this, new ConstantDoubleVertex(0.5));
    }

    @Override
    default LogVertex log() {
        return new LogVertex(this);
    }

    @Override
    default LogGammaVertex logGamma() {
        return new LogGammaVertex(this);
    }

    @Override
    default DoubleVertex digamma() {
        return null;
    }

    @Override
    default SigmoidVertex sigmoid() {
        return new SigmoidVertex(this);
    }

    @Override
    default DoubleVertex choleskyDecomposition() {
        return null;
    }

    @Override
    default SinVertex sin() {
        return new SinVertex(this);
    }

    @Override
    default CosVertex cos() {
        return new CosVertex(this);
    }

    @Override
    default TanVertex tan() {
        return new TanVertex(this);
    }

    @Override
    default ArcSinVertex asin() {
        return new ArcSinVertex(this);
    }

    @Override
    default ArcCosVertex acos() {
        return new ArcCosVertex(this);
    }

    @Override
    default DoubleVertex sinh() {
        return null;
    }

    @Override
    default DoubleVertex cosh() {
        return null;
    }

    @Override
    default DoubleVertex tanh() {
        return null;
    }

    @Override
    default DoubleVertex asinh() {
        return null;
    }

    @Override
    default DoubleVertex acosh() {
        return null;
    }

    @Override
    default DoubleVertex atanh() {
        return null;
    }

    @Override
    default ArcTanVertex atan() {
        return new ArcTanVertex(this);
    }

    @Override
    default ArcTan2Vertex atan2(Double y) {
        return atan2(new ConstantDoubleVertex(y));
    }

    @Override
    default ArcTan2Vertex atan2(DoubleVertex that) {
        return new ArcTan2Vertex(this, that);
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
    default DoubleVertex matrixMultiply(DoubleVertex that) {
        int leftRank = this.getRank();
        int rightRank = that.getRank();

        if (leftRank < 1 || rightRank < 1) {
            throw new IllegalArgumentException("Matrix multiply for rank 0 is not supported. Use times instead.");
        }

        DoubleVertex leftMatrix = leftRank == 1 ? this.reshape(1, this.getShape()[0]) : this;
        DoubleVertex rightMatrix = rightRank == 1 ? that.reshape(that.getShape()[0], 1) : that;

        MatrixMultiplicationVertex result = new MatrixMultiplicationVertex(leftMatrix, rightMatrix);

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
    default DoubleVertex tensorMultiply(DoubleVertex value, int[] dimLeft, int[] dimsRight) {
        return null;
    }

    @Override
    default MatrixInverseVertex matrixInverse() {
        return new MatrixInverseVertex(this);
    }

    @Override
    default DoubleVertex standardize() {
        return null;
    }

    default MatrixDeterminantVertex matrixDeterminant() {
        return determinant();
    }

    @Override
    default MatrixDeterminantVertex determinant() {
        return new MatrixDeterminantVertex(this);
    }

    default DoubleUnaryOpLambda<DoubleTensor> lambda(long[] outputShape, Function<DoubleTensor, DoubleTensor> op,
                                                     Function<Map<IVertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                                     Function<PartialDerivative, Map<IVertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(outputShape, this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    default DoubleUnaryOpLambda<DoubleTensor> lambda(Function<DoubleTensor, DoubleTensor> op,
                                                     Function<Map<IVertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                                     Function<PartialDerivative, Map<IVertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    @Override
    default BooleanVertex toBoolean() {
        return null;
    }

    @Override
    default DoubleVertex toDouble() {
        return null;
    }

    @Override
    default IntegerVertex toInteger() {
        return new CastToIntegerVertex(this);
    }
}
