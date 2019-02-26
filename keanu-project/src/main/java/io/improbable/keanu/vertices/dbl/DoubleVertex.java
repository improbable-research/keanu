package io.improbable.keanu.vertices.dbl;


import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.network.NetworkLoader;
import io.improbable.keanu.network.NetworkSaver;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
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

import java.util.Map;
import java.util.function.Function;

public abstract class DoubleVertex extends Vertex<DoubleTensor> implements DoubleOperators<DoubleVertex> {

    public DoubleVertex(long[] initialShape) {
        super(initialShape);
    }

    /**
     * @param dimension dimension to concat along. Negative dimension indexing is not supported.
     * @param toConcat  array of things to concat. Must match in all dimensions except for the provided
     *                  dimension
     * @return a vertex that represents the concatenation of the toConcat
     */
    public static ConcatenationVertex concat(int dimension, DoubleVertex... toConcat) {
        return new ConcatenationVertex(dimension, toConcat);
    }

    public static MinVertex min(DoubleVertex a, DoubleVertex b) {
        return new MinVertex(a, b);
    }

    public static MaxVertex max(DoubleVertex a, DoubleVertex b) {
        return new MaxVertex(a, b);
    }

    public DifferenceVertex minus(DoubleVertex that) {
        return new DifferenceVertex(this, that);
    }

    public AdditionVertex plus(DoubleVertex that) {
        return new AdditionVertex(this, that);
    }

    public MultiplicationVertex multiply(DoubleVertex that) {
        return new MultiplicationVertex(this, that);
    }

    public DoubleVertex matrixMultiply(DoubleVertex that) {
        int leftRank = this.getRank();
        int rightRank = that.getRank();

        DoubleVertex leftMatrix = leftRank == 1 ? this.reshape(1, this.getShape()[0]) : this;
        DoubleVertex rightMatrix = rightRank == 1 ? that.reshape(that.getShape()[0], 1) : that;

        MatrixMultiplicationVertex result = new MatrixMultiplicationVertex(leftMatrix, rightMatrix);

        if (leftRank == 1 && rightRank == 1) {
            return result.reshape();
        } else if (leftRank == 1) {
            return result.reshape(result.getShape()[1]);
        } else if (rightRank == 1) {
            return result.reshape(result.getShape()[0]);
        } else {
            return result;
        }
    }

    public MatrixInverseVertex matrixInverse() {
        return new MatrixInverseVertex(this);
    }

    public MatrixDeterminantVertex matrixDeterminant() {
        return new MatrixDeterminantVertex(this);
    }

    public DivisionVertex divideBy(DoubleVertex that) {
        return new DivisionVertex(this, that);
    }

    public PowerVertex pow(DoubleVertex exponent) {
        return new PowerVertex(this, exponent);
    }

    @Override
    public DifferenceVertex minus(double that) {
        return minus(new ConstantDoubleVertex(that));
    }

    @Override
    public DifferenceVertex reverseMinus(double that) {
        return new ConstantDoubleVertex(that).minus(this);
    }

    @Override
    public AdditionVertex plus(double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    public MultiplicationVertex multiply(double that) {
        return multiply(new ConstantDoubleVertex(that));
    }

    public DivisionVertex divideBy(double that) {
        return divideBy(new ConstantDoubleVertex(that));
    }

    @Override
    public PowerVertex pow(double that) {
        return pow(new ConstantDoubleVertex(that));
    }

    public AbsVertex abs() {
        return new AbsVertex(this);
    }

    public FloorVertex floor() {
        return new FloorVertex(this);
    }

    public CeilVertex ceil() {
        return new CeilVertex(this);
    }

    public RoundVertex round() {
        return new RoundVertex(this);
    }

    @Override
    public ExpVertex exp() {
        return new ExpVertex(this);
    }

    public LogVertex log() {
        return new LogVertex(this);
    }

    public LogGammaVertex logGamma() {
        return new LogGammaVertex(this);
    }

    public SigmoidVertex sigmoid() {
        return new SigmoidVertex(this);
    }

    @Override
    public SinVertex sin() {
        return new SinVertex(this);
    }

    @Override
    public CosVertex cos() {
        return new CosVertex(this);
    }

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

    public ArcTanVertex atan() {
        return new ArcTanVertex(this);
    }

    public ArcTan2Vertex atan2(DoubleVertex that) {
        return new ArcTan2Vertex(this, that);
    }

    /**
     * Sum over all dimensions. This will always result in a scalar.
     *
     * @return a vertex representing the summation result
     */
    public SumVertex sum() {
        return new SumVertex(this);
    }

    /**
     * Sum over specified dimensions.
     *
     * @param sumOverDimensions dimensions to sum over. Negative dimension indexing is not supported
     * @return a vertex representing the summation result
     */
    public SumVertex sum(int... sumOverDimensions) {
        return new SumVertex(this, sumOverDimensions);
    }

    public ReshapeVertex reshape(long... proposedShape) {
        return new ReshapeVertex(this, proposedShape);
    }

    public PermuteVertex permute(int... rearrange) {
        return new PermuteVertex(this, rearrange);
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

    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)
    @Override
    public MultiplicationVertex times(DoubleVertex that) {
        return multiply(that);
    }

    @Override
    public DivisionVertex div(DoubleVertex that) {
        return divideBy(that);
    }

    @Override
    public MultiplicationVertex times(double that) {
        return multiply(that);
    }

    @Override
    public DivisionVertex div(double that) {
        return divideBy(that);
    }

    @Override
    public DivisionVertex reverseDiv(double that) {
        return new ConstantDoubleVertex(that).div(this);
    }

    @Override
    public MultiplicationVertex unaryMinus() {
        return multiply(-1.0);
    }

    public BooleanVertex equalTo(DoubleVertex rhs) {
        return new EqualsVertex<>(this, rhs);
    }

    public IntegerVertex toInteger() { return new CastToIntegerVertex(this); }

    public <T extends Tensor> BooleanVertex notEqualTo(Vertex<T> rhs) {
        return new NotEqualsVertex<>(this, rhs);
    }

    public <T extends NumberTensor> BooleanVertex greaterThan(Vertex<T> rhs) {
        return new GreaterThanVertex<>(this, rhs);
    }

    public DoubleVertex toGreaterThanMask(DoubleVertex rhs) {
        return new DoubleGreaterThanMaskVertex(this, rhs);
    }

    public DoubleVertex toGreaterThanMask(double rhs) {
        return toGreaterThanMask(new ConstantDoubleVertex(rhs));
    }

    public <T extends NumberTensor> BooleanVertex greaterThanOrEqualTo(Vertex<T> rhs) {
        return new GreaterThanOrEqualVertex<>(this, rhs);
    }

    public DoubleVertex toGreaterThanOrEqualToMask(DoubleVertex rhs) {
        return new DoubleGreaterThanOrEqualToMaskVertex(this, rhs);
    }

    public DoubleVertex toGreaterThanOrEqualToMask(double rhs) {
        return toGreaterThanOrEqualToMask(new ConstantDoubleVertex(rhs));
    }

    public <T extends NumberTensor> BooleanVertex lessThan(Vertex<T> rhs) {
        return new LessThanVertex<>(this, rhs);
    }

    public DoubleVertex toLessThanMask(DoubleVertex rhs) {
        return new DoubleLessThanMaskVertex(this, rhs);
    }

    public DoubleVertex toLessThanMask(double rhs) {
        return toLessThanMask(new ConstantDoubleVertex(rhs));
    }

    public <T extends NumberTensor> BooleanVertex lessThanOrEqualTo(Vertex<T> rhs) {
        return new LessThanOrEqualVertex<>(this, rhs);
    }

    public DoubleVertex toLessThanOrEqualToMask(DoubleVertex rhs) {
        return new DoubleLessThanOrEqualToMaskVertex(this, rhs);
    }

    public DoubleVertex toLessThanOrEqualToMask(double rhs) {
        return toLessThanOrEqualToMask(new ConstantDoubleVertex(rhs));
    }

    public DoubleVertex setWithMask(DoubleVertex mask, double value) {
        return setWithMask(mask, new ConstantDoubleVertex(value));
    }

    public DoubleVertex setWithMask(DoubleVertex mask, DoubleVertex value) {
        return new DoubleSetWithMaskVertex(this, mask, value);
    }

    public TakeVertex take(long... index) {
        return new TakeVertex(this, index);
    }

    public SliceVertex slice(int dimension, int index) {
        return new SliceVertex(this, dimension, index);
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
        super.observe(DoubleTensor.scalar(value));
    }

    public void observe(double[] values) {
        super.observe(DoubleTensor.create(values));
    }

    public double getValue(int... index) {
        return getValue().getValue(index);
    }

    @Override
    public void loadValue(NetworkLoader loader) {
        loader.loadValue(this);
    }

    public void saveValue(NetworkSaver netSaver) {
        netSaver.saveValue(this);
    }
}
