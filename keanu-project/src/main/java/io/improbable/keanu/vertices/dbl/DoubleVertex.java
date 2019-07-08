package io.improbable.keanu.vertices.dbl;


import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.FloatingPointTensorVertex;
import io.improbable.keanu.vertices.Vertex;
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

import java.util.List;
import java.util.Map;
import java.util.function.Function;

public abstract class DoubleVertex extends Vertex<DoubleTensor> implements DoubleOperators<DoubleVertex>, FloatingPointTensorVertex<Double, DoubleVertex> {

    public DoubleVertex(long[] initialShape) {
        super(initialShape);
    }

    //////////////////////////
    ////  Vertex helpers
    //////////////////////////

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
        super.observe(DoubleTensor.scalar(value));
    }

    public void observe(double[] values) {
        super.observe(DoubleTensor.create(values));
    }

    public double getValue(long... index) {
        return getValue().getValue(index);
    }

    @Override
    public void loadValue(NetworkLoader loader) {
        loader.loadValue(this);
    }

    @Override
    public void saveValue(NetworkSaver netSaver) {
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
    public static ConcatenationVertex concat(int dimension, DoubleVertex... toConcat) {
        return new ConcatenationVertex(dimension, toConcat);
    }

    @Override
    public ReshapeVertex reshape(long... proposedShape) {
        return new ReshapeVertex(this, proposedShape);
    }

    @Override
    public PermuteVertex permute(int... rearrange) {
        return new PermuteVertex(this, rearrange);
    }

    @Override
    public DoubleVertex broadcast(long... toShape) {
        return null;
    }

    @Override
    public PermuteVertex transpose() {
        return new PermuteVertex(this, 1, 0);
    }

    @Override
    public TakeVertex take(long... index) {
        return new TakeVertex(this, index);
    }

    @Override
    public List<DoubleVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    @Override
    public DoubleVertex diag() {
        return null;
    }

    @Override
    public DoubleVertex get(BooleanVertex booleanIndex) {
        return null;
    }

    @Override
    public SliceVertex slice(int dimension, long index) {
        return new SliceVertex(this, dimension, index);
    }

    @Override
    public DoubleVertex slice(Slicer slicer) {
        return null;
    }

    @Override
    public BooleanVertex elementwiseEquals(DoubleVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    @Override
    public BooleanVertex elementwiseEquals(Double value) {
        return new EqualsVertex<>(this, new ConstantDoubleVertex(value));
    }

    public <T extends Tensor> BooleanVertex notEqualTo(Vertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    //////////////////////////
    ////  Number Tensor Operations
    //////////////////////////

    @Override
    public SumVertex sum() {
        return new SumVertex(this);
    }

    @Override
    public SumVertex sum(int... sumOverDimensions) {
        return new SumVertex(this, sumOverDimensions);
    }

    @Override
    public DoubleVertex cumSum(int requestedDimension) {
        return null;
    }

    @Override
    public DoubleVertex product() {
        return null;
    }

    @Override
    public DoubleVertex product(int... overDimensions) {
        return null;
    }

    @Override
    public DoubleVertex cumProd(int requestedDimension) {
        return null;
    }

    public static MinVertex min(DoubleVertex a, DoubleVertex b) {
        return new MinVertex(a, b);
    }

    public static MaxVertex max(DoubleVertex a, DoubleVertex b) {
        return new MaxVertex(a, b);
    }

    @Override
    public DoubleVertex max() {
        return null;
    }

    @Override
    public DoubleVertex max(DoubleVertex that) {
        return max(this, that);
    }

    @Override
    public DoubleVertex min() {
        return null;
    }

    @Override
    public DoubleVertex min(DoubleVertex that) {
        return min(this, that);
    }

    @Override
    public DoubleVertex clamp(DoubleVertex min, DoubleVertex max) {
        return null;
    }

    @Override
    public DifferenceVertex minus(double that) {
        return minus(new ConstantDoubleVertex(that));
    }

    @Override
    public DifferenceVertex minus(DoubleVertex that) {
        return new DifferenceVertex(this, that);
    }

    @Override
    public DoubleVertex minus(Double value) {
        return null;
    }

    @Override
    public MultiplicationVertex unaryMinus() {
        return multiply(-1.0);
    }

    @Override
    public DoubleVertex reverseMinus(DoubleVertex value) {
        return null;
    }

    @Override
    public DoubleVertex reverseMinus(Double value) {
        return null;
    }

    @Override
    public DifferenceVertex reverseMinus(double that) {
        return new ConstantDoubleVertex(that).minus(this);
    }

    @Override
    public DoubleVertex plus(double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    @Override
    public AdditionVertex plus(Double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    @Override
    public AdditionVertex plus(DoubleVertex that) {
        return new AdditionVertex(this, that);
    }

    public MultiplicationVertex multiply(double that) {
        return multiply(new ConstantDoubleVertex(that));
    }

    public MultiplicationVertex multiply(DoubleVertex that) {
        return new MultiplicationVertex(this, that);
    }

    @Override
    public MultiplicationVertex times(DoubleVertex that) {
        return multiply(that);
    }

    @Override
    public DoubleVertex times(Double value) {
        return multiply(value);
    }

    @Override
    public MultiplicationVertex times(double that) {
        return multiply(that);
    }

    @Override
    public DoubleVertex div(Double value) {
        return divideBy(value);
    }

    public DivisionVertex divideBy(double that) {
        return divideBy(new ConstantDoubleVertex(that));
    }

    public DivisionVertex divideBy(DoubleVertex that) {
        return new DivisionVertex(this, that);
    }

    @Override
    public DivisionVertex div(DoubleVertex that) {
        return divideBy(that);
    }

    @Override
    public DivisionVertex div(double that) {
        return divideBy(that);
    }

    @Override
    public DoubleVertex reverseDiv(Double value) {
        return null;
    }

    @Override
    public DoubleVertex reverseDiv(DoubleVertex value) {
        return null;
    }

    @Override
    public DivisionVertex reverseDiv(double that) {
        return new ConstantDoubleVertex(that).div(this);
    }

    @Override
    public PowerVertex pow(double that) {
        return pow(new ConstantDoubleVertex(that));
    }

    @Override
    public PowerVertex pow(DoubleVertex exponent) {
        return new PowerVertex(this, exponent);
    }

    @Override
    public DoubleVertex pow(Double exponent) {
        return null;
    }

    @Override
    public DoubleVertex average() {
        return null;
    }

    @Override
    public DoubleVertex standardDeviation() {
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
    public AbsVertex abs() {
        return new AbsVertex(this);
    }

    @Override
    public BooleanVertex greaterThan(DoubleVertex rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    @Override
    public BooleanVertex greaterThanOrEqual(DoubleVertex rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    @Override
    public BooleanVertex greaterThan(Double value) {
        return new GreaterThanVertex<>(this, new ConstantDoubleVertex(value));
    }

    @Override
    public BooleanVertex greaterThanOrEqual(Double value) {
        return new GreaterThanOrEqualVertex<>(this, new ConstantDoubleVertex(value));
    }

    @Override
    public BooleanVertex lessThan(Double value) {
        return new LessThanVertex<>(this, new ConstantDoubleVertex(value));
    }

    @Override
    public BooleanVertex lessThanOrEqual(Double value) {
        return new LessThanOrEqualVertex<>(this, new ConstantDoubleVertex(value));
    }

    @Override
    public BooleanVertex lessThan(DoubleVertex rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    @Override
    public BooleanVertex lessThanOrEqual(DoubleVertex rhs) {
        return new LessThanOrEqualVertex<>(this, rhs);
    }

    @Override
    public DoubleVertex greaterThanMask(DoubleVertex rhs) {
        return new DoubleGreaterThanMaskVertex(this, rhs);
    }

    public DoubleVertex greaterThanMask(Double rhs) {
        return greaterThanMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    public DoubleVertex greaterThanOrEqualToMask(DoubleVertex rhs) {
        return new DoubleGreaterThanOrEqualToMaskVertex(this, rhs);
    }

    public DoubleVertex greaterThanOrEqualToMask(Double rhs) {
        return greaterThanOrEqualToMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    public DoubleVertex lessThanMask(DoubleVertex rhs) {
        return new DoubleLessThanMaskVertex(this, rhs);
    }

    public DoubleVertex lessThanMask(Double rhs) {
        return lessThanMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    public DoubleVertex lessThanOrEqualToMask(DoubleVertex rhs) {
        return new DoubleLessThanOrEqualToMaskVertex(this, rhs);
    }

    public DoubleVertex lessThanOrEqualToMask(Double rhs) {
        return lessThanOrEqualToMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    public DoubleVertex setWithMask(DoubleVertex mask, Double value) {
        return setWithMask(mask, new ConstantDoubleVertex(value));
    }

    @Override
    public DoubleVertex apply(Function<Double, Double> function) {
        return null;
    }

    @Override
    public DoubleVertex safeLogTimes(DoubleVertex y) {
        return null;
    }

    @Override
    public BooleanVertex equalsWithinEpsilon(DoubleVertex other, Double epsilon) {
        return new NumericalEqualsVertex<>(this, other, epsilon);
    }

    public DoubleVertex setWithMask(DoubleVertex mask, DoubleVertex value) {
        return new DoubleSetWithMaskVertex(this, mask, value);
    }

    //////////////////////////
    ////  Floating Point Tensor Operations
    //////////////////////////

    @Override
    public FloorVertex floor() {
        return new FloorVertex(this);
    }

    @Override
    public CeilVertex ceil() {
        return new CeilVertex(this);
    }

    @Override
    public RoundVertex round() {
        return new RoundVertex(this);
    }

    @Override
    public ExpVertex exp() {
        return new ExpVertex(this);
    }

    @Override
    public DoubleVertex logAddExp2(DoubleVertex that) {
        return null;
    }

    @Override
    public DoubleVertex logAddExp(DoubleVertex that) {
        return null;
    }

    @Override
    public DoubleVertex log1p() {
        return null;
    }

    @Override
    public DoubleVertex log2() {
        return null;
    }

    @Override
    public DoubleVertex log10() {
        return null;
    }

    @Override
    public DoubleVertex exp2() {
        return null;
    }

    @Override
    public DoubleVertex expM1() {
        return null;
    }

    @Override
    public DoubleVertex replaceNaN(Double value) {
        return null;
    }

    @Override
    public BooleanVertex notNaN() {
        return null;
    }

    @Override
    public BooleanVertex isNaN() {
        return null;
    }

    @Override
    public BooleanVertex isFinite() {
        return null;
    }

    @Override
    public BooleanVertex isInfinite() {
        return null;
    }

    @Override
    public BooleanVertex isNegativeInfinity() {
        return null;
    }

    @Override
    public BooleanVertex isPositiveInfinity() {
        return null;
    }

    @Override
    public IntegerVertex nanArgMax(int axis) {
        return null;
    }

    @Override
    public IntegerVertex nanArgMax() {
        return null;
    }

    @Override
    public IntegerVertex nanArgMin(int axis) {
        return null;
    }

    @Override
    public IntegerVertex nanArgMin() {
        return null;
    }

    @Override
    public DoubleVertex reciprocal() {
        return null;
    }

    @Override
    public DoubleVertex sqrt() {
        return new PowerVertex(this, new ConstantDoubleVertex(0.5));
    }

    @Override
    public LogVertex log() {
        return new LogVertex(this);
    }

    @Override
    public LogGammaVertex logGamma() {
        return new LogGammaVertex(this);
    }

    @Override
    public DoubleVertex digamma() {
        return null;
    }

    @Override
    public SigmoidVertex sigmoid() {
        return new SigmoidVertex(this);
    }

    @Override
    public DoubleVertex choleskyDecomposition() {
        return null;
    }

    @Override
    public SinVertex sin() {
        return new SinVertex(this);
    }

    @Override
    public CosVertex cos() {
        return new CosVertex(this);
    }

    @Override
    public TanVertex tan() {
        return new TanVertex(this);
    }

    @Override
    public ArcSinVertex asin() {
        return new ArcSinVertex(this);
    }

    @Override
    public ArcCosVertex acos() {
        return new ArcCosVertex(this);
    }

    @Override
    public DoubleVertex sinh() {
        return null;
    }

    @Override
    public DoubleVertex cosh() {
        return null;
    }

    @Override
    public DoubleVertex tanh() {
        return null;
    }

    @Override
    public DoubleVertex asinh() {
        return null;
    }

    @Override
    public DoubleVertex acosh() {
        return null;
    }

    @Override
    public DoubleVertex atanh() {
        return null;
    }

    @Override
    public ArcTanVertex atan() {
        return new ArcTanVertex(this);
    }

    @Override
    public DoubleVertex atan2(Double y) {
        return atan2(new ConstantDoubleVertex(y));
    }

    @Override
    public ArcTan2Vertex atan2(DoubleVertex that) {
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
    public DoubleVertex matrixMultiply(DoubleVertex that) {
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
    public DoubleVertex tensorMultiply(DoubleVertex value, int[] dimLeft, int[] dimsRight) {
        return null;
    }

    @Override
    public MatrixInverseVertex matrixInverse() {
        return new MatrixInverseVertex(this);
    }

    @Override
    public DoubleVertex standardize() {
        return null;
    }

    public MatrixDeterminantVertex matrixDeterminant() {
        return determinant();
    }

    @Override
    public MatrixDeterminantVertex determinant() {
        return new MatrixDeterminantVertex(this);
    }

    public DoubleUnaryOpLambda<DoubleTensor> lambda(long[] outputShape, Function<DoubleTensor, DoubleTensor> op,
                                                    Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                                    Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(outputShape, this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    public DoubleUnaryOpLambda<DoubleTensor> lambda(Function<DoubleTensor, DoubleTensor> op,
                                                    Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                                    Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    @Override
    public BooleanVertex toBoolean() {
        return null;
    }

    @Override
    public DoubleVertex toDouble() {
        return null;
    }

    @Override
    public IntegerVertex toInteger() {
        return new CastToIntegerVertex(this);
    }
}
