package io.improbable.keanu.vertices.dbl;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
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
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.SinVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.TanVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
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
    default DoubleVertex wrap(NonProbabilisticVertex<DoubleTensor, DoubleVertex> vertex) {
        return new DoubleVertexWrapper(vertex);
    }

    @Override
    default Class<?> ofType() {
        return DoubleTensor.class;
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
    static DoubleVertex concat(int dimension, DoubleVertex... toConcat) {
        return new ConcatenationVertex(dimension, toConcat);
    }

    @Override
    default List<DoubleVertex> split(int dimension, long... splitAtIndices) {
        return null;
    }

    @Override
    default DoubleVertex slice(Slicer slicer) {
        return null;
    }

    //////////////////////////
    ////  Number Tensor Operations
    //////////////////////////

    static DoubleVertex min(DoubleVertex a, DoubleVertex b) {
        return a.min(b);
    }

    static DoubleVertex max(DoubleVertex a, DoubleVertex b) {
        return a.max(b);
    }

    @Override
    default DoubleVertex clamp(DoubleVertex min, DoubleVertex max) {
        return null;
    }

    @Override
    default DoubleVertex minus(double that) {
        return minus(new ConstantDoubleVertex(that));
    }

    @Override
    default DoubleVertex minus(Double value) {
        return minus(new ConstantDoubleVertex(value));
    }

    @Override
    default DoubleVertex unaryMinus() {
        return multiply(-1.0);
    }

    @Override
    default DoubleVertex reverseMinus(Double value) {
        return reverseDiv(new ConstantDoubleVertex(value));
    }

    @Override
    default DoubleVertex reverseMinus(double that) {
        return new ConstantDoubleVertex(that).minus(this);
    }

    @Override
    default DoubleVertex plus(double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    @Override
    default DoubleVertex plus(Double that) {
        return plus(new ConstantDoubleVertex(that));
    }

    default DoubleVertex multiply(DoubleVertex that) {
        return times(that);
    }

    default DoubleVertex multiply(Double that) {
        return times(new ConstantDoubleVertex(that));
    }

    default DoubleVertex multiply(double that) {
        return times(new ConstantDoubleVertex(that));
    }

    @Override
    default DoubleVertex times(Double value) {
        return times(new ConstantDoubleVertex(value));
    }

    @Override
    default DoubleVertex times(double that) {
        return times(new ConstantDoubleVertex(that));
    }

    default DoubleVertex divideBy(DoubleVertex that) {
        return div(that);
    }

    default DoubleVertex divideBy(Double that) {
        return div(new ConstantDoubleVertex(that));
    }

    default DoubleVertex divideBy(double that) {
        return div(new ConstantDoubleVertex(that));
    }

    @Override
    default DoubleVertex div(Double value) {
        return div(new ConstantDoubleVertex(value));
    }

    @Override
    default DoubleVertex div(double that) {
        return div(new ConstantDoubleVertex(that));
    }

    @Override
    default DoubleVertex reverseDiv(Double value) {
        return reverseDiv(new ConstantDoubleVertex(value));
    }

    @Override
    default DoubleVertex reverseDiv(double value) {
        return reverseDiv(new ConstantDoubleVertex(value));
    }

    @Override
    default DoubleVertex pow(double that) {
        return pow(new ConstantDoubleVertex(that));
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
    default DoubleVertex standardize() {
        return null;
    }

    default DoubleVertex greaterThanMask(Double rhs) {
        return greaterThanMask(new ConstantDoubleVertex(rhs));
    }

    default DoubleVertex greaterThanOrEqualToMask(Double rhs) {
        return greaterThanOrEqualToMask(new ConstantDoubleVertex(rhs));
    }

    default DoubleVertex lessThanMask(Double rhs) {
        return lessThanMask(new ConstantDoubleVertex(rhs));
    }

    default DoubleVertex lessThanOrEqualToMask(Double rhs) {
        return lessThanOrEqualToMask(new ConstantDoubleVertex(rhs));
    }

    @Override
    default DoubleVertex safeLogTimes(DoubleVertex y) {
        return null;
    }

    //////////////////////////
    ////  Floating Point Tensor Operations
    //////////////////////////

    @Override
    default DoubleVertex floor() {
        return new FloorVertex(this);
    }

    @Override
    default DoubleVertex ceil() {
        return new CeilVertex(this);
    }

    @Override
    default DoubleVertex round() {
        return new RoundVertex(this);
    }

    @Override
    default DoubleVertex exp() {
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
        return reverseDiv(1.0);
    }

    @Override
    default DoubleVertex sqrt() {
        return pow(new ConstantDoubleVertex(0.5));
    }

    @Override
    default DoubleVertex log() {
        return new LogVertex(this);
    }

    @Override
    default DoubleVertex logGamma() {
        return new LogGammaVertex(this);
    }

    @Override
    default DoubleVertex digamma() {
        return null;
    }

    @Override
    default DoubleVertex sigmoid() {
        return new SigmoidVertex(this);
    }

    @Override
    default DoubleVertex choleskyDecomposition() {
        return null;
    }

    @Override
    default DoubleVertex sin() {
        return new SinVertex(this);
    }

    @Override
    default DoubleVertex cos() {
        return new CosVertex(this);
    }

    @Override
    default DoubleVertex tan() {
        return new TanVertex(this);
    }

    @Override
    default DoubleVertex asin() {
        return new ArcSinVertex(this);
    }

    @Override
    default DoubleVertex acos() {
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
    default DoubleVertex atan() {
        return new ArcTanVertex(this);
    }

    @Override
    default DoubleVertex atan2(Double y) {
        return atan2(new ConstantDoubleVertex(y));
    }

    @Override
    default DoubleVertex atan2(DoubleVertex that) {
        return new ArcTan2Vertex(this, that);
    }

    @Override
    default DoubleVertex matrixInverse() {
        return new MatrixInverseVertex(this);
    }


    default DoubleVertex matrixDeterminant() {
        return determinant();
    }

    @Override
    default DoubleVertex determinant() {
        return new MatrixDeterminantVertex(this);
    }

    default DoubleUnaryOpLambda<DoubleTensor> lambda(long[] outputShape, Function<DoubleTensor, DoubleTensor> op,
                                                     Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                                     Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(outputShape, this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    default DoubleUnaryOpLambda<DoubleTensor> lambda(Function<DoubleTensor, DoubleTensor> op,
                                                     Function<Map<Vertex, PartialDerivative>, PartialDerivative> forwardModeAutoDiffLambda,
                                                     Function<PartialDerivative, Map<Vertex, PartialDerivative>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

}
