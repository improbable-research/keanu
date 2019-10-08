package io.improbable.keanu.vertices.tensor.number.floating.dbl;

import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.FloatingPointTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ForwardModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.ReverseModePartialDerivative;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.multiple.ConcatenationVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CosVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SinVertex;

import java.util.Map;
import java.util.function.Function;

public interface DoubleVertex extends DoubleOperators<DoubleVertex>, FloatingPointTensorVertex<Double, DoubleTensor, DoubleVertex> {

    //////////////////////////
    ////  Vertex helpersx
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

    //////////////////////////
    ////  Floating Point Tensor Operations
    //////////////////////////

    @Override
    default DoubleVertex log() {
        return wrap(new LogVertex<>(this));
    }

    @Override
    default DoubleVertex exp() {
        return wrap(new ExpVertex<>(this));
    }

    @Override
    default DoubleVertex sin() {
        return wrap(new SinVertex<>(this));
    }

    @Override
    default DoubleVertex cos() {
        return wrap(new CosVertex<>(this));
    }

    @Override
    default DoubleVertex asin() {
        return wrap(new ArcSinVertex<>(this));
    }

    @Override
    default DoubleVertex acos() {
        return wrap(new ArcCosVertex<>(this));
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
    default DoubleVertex atan2(Double y) {
        return atan2(new ConstantDoubleVertex(y));
    }

    default DoubleUnaryOpLambda<DoubleTensor> lambda(long[] outputShape, Function<DoubleTensor, DoubleTensor> op,
                                                     Function<Map<Vertex, ForwardModePartialDerivative>, ForwardModePartialDerivative> forwardModeAutoDiffLambda,
                                                     Function<ReverseModePartialDerivative, Map<Vertex, ReverseModePartialDerivative>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(outputShape, this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

    default DoubleUnaryOpLambda<DoubleTensor> lambda(Function<DoubleTensor, DoubleTensor> op,
                                                     Function<Map<Vertex, ForwardModePartialDerivative>, ForwardModePartialDerivative> forwardModeAutoDiffLambda,
                                                     Function<ReverseModePartialDerivative, Map<Vertex, ReverseModePartialDerivative>> reverseModeAutoDiffLambda) {
        return new DoubleUnaryOpLambda<>(this, op, forwardModeAutoDiffLambda, reverseModeAutoDiffLambda);
    }

}
