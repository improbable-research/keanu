package io.improbable.keanu.vertices.tensor.number.floating;

import io.improbable.keanu.BaseFloatingPointTensor;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.IsFiniteVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.IsInfiniteVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.IsNaNVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.IsNegativeInfinityVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.IsPositiveInfinityVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary.NotNaNVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.NaNArgMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.NaNArgMinVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.binary.ArcTan2Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.binary.LogAddExp2Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.binary.LogAddExpVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcCoshVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcTanhVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcSinhVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcTanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CholeskyDecompositionVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CosVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CoshVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.DigammaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Exp2Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ExpM1Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log10Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log1pVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.Log2Vertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogGammaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MatrixDeterminantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MatrixInverseVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MeanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ReplaceNaNVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SinVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SinhVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.StandardDeviationVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.StandardizeVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TanhVertex;

public interface FloatingPointTensorVertex<T extends Number, TENSOR extends FloatingPointTensor<T, TENSOR>, VERTEX extends FloatingPointTensorVertex<T, TENSOR, VERTEX>>
    extends NumberTensorVertex<T, TENSOR, VERTEX>, BaseFloatingPointTensor<BooleanVertex, IntegerVertex, DoubleVertex, T, VERTEX> {

    @Override
    default VERTEX ceil() {
        return wrap(new CeilVertex<>(this));
    }

    @Override
    default VERTEX round() {
        return wrap(new RoundVertex<>(this));
    }

    @Override
    default VERTEX floor() {
        return wrap(new FloorVertex<>(this));
    }

    @Override
    default VERTEX sin() {
        return wrap(new SinVertex<>(this));
    }

    @Override
    default VERTEX cos() {
        return wrap(new CosVertex<>(this));
    }

    @Override
    default VERTEX tan() {
        return wrap(new TanVertex<>(this));
    }

    @Override
    default VERTEX asin() {
        return wrap(new ArcSinVertex<>(this));
    }

    @Override
    default VERTEX acos() {
        return wrap(new ArcCosVertex<>(this));
    }

    @Override
    default VERTEX mean() {
        return wrap(new MeanVertex<>(this));
    }

    @Override
    default VERTEX standardDeviation() {
        return wrap(new StandardDeviationVertex<>(this));
    }

    @Override
    default VERTEX standardize() {
        return wrap(new StandardizeVertex<>(this));
    }

    @Override
    default VERTEX logAddExp2(VERTEX that) {
        return wrap(new LogAddExp2Vertex<>(this, that));
    }

    @Override
    default VERTEX logAddExp(VERTEX that) {
        return wrap(new LogAddExpVertex<>(this, that));
    }

    @Override
    default VERTEX log1p() {
        return wrap(new Log1pVertex<>(this));
    }

    @Override
    default VERTEX log2() {
        return wrap(new Log2Vertex<>(this));
    }

    @Override
    default VERTEX log10() {
        return wrap(new Log10Vertex<>(this));
    }

    @Override
    default VERTEX exp2() {
        return wrap(new Exp2Vertex<>(this));
    }

    @Override
    default VERTEX expM1() {
        return wrap(new ExpM1Vertex<>(this));
    }

    @Override
    default VERTEX replaceNaN(T value) {
        return wrap(new ReplaceNaNVertex<>(this, value));
    }

    @Override
    default BooleanVertex notNaN() {
        return new NotNaNVertex<>(this);
    }

    @Override
    default BooleanVertex isNaN() {
        return new IsNaNVertex<>(this);
    }

    @Override
    default BooleanVertex isFinite() {
        return new IsFiniteVertex<>(this);
    }

    @Override
    default BooleanVertex isInfinite() {
        return new IsInfiniteVertex<>(this);
    }

    @Override
    default BooleanVertex isNegativeInfinity() {
        return new IsNegativeInfinityVertex<>(this);
    }

    @Override
    default BooleanVertex isPositiveInfinity() {
        return new IsPositiveInfinityVertex<>(this);
    }

    @Override
    default VERTEX exp() {
        return wrap(new ExpVertex<>(this));
    }


    @Override
    default VERTEX log() {
        return wrap(new LogVertex<>(this));
    }

    @Override
    default VERTEX logGamma() {
        return wrap(new LogGammaVertex<>(this));
    }

    @Override
    default VERTEX digamma() {
        return wrap(new DigammaVertex<>(this));
    }

    @Override
    default VERTEX sigmoid() {
        return wrap(new SigmoidVertex<>(this));
    }

    @Override
    default VERTEX choleskyDecomposition() {
        return wrap(new CholeskyDecompositionVertex<>(this));
    }

    @Override
    default VERTEX sinh() {
        return wrap(new SinhVertex<>(this));
    }

    @Override
    default VERTEX cosh() {
        return wrap(new CoshVertex<>(this));
    }

    @Override
    default VERTEX tanh() {
        return wrap(new TanhVertex<>(this));
    }

    @Override
    default VERTEX asinh() {
        return wrap(new ArcSinhVertex<>(this));
    }

    @Override
    default VERTEX acosh() {
        return wrap(new ArcCoshVertex<>(this));
    }

    @Override
    default VERTEX atanh() {
        return wrap(new ArcTanhVertex<>(this));
    }

    @Override
    default VERTEX atan() {
        return wrap(new ArcTanVertex<>(this));
    }

    @Override
    default VERTEX atan2(VERTEX that) {
        return wrap(new ArcTan2Vertex<>(this, that));
    }

    @Override
    default VERTEX matrixInverse() {
        return wrap(new MatrixInverseVertex<>(this));
    }

    @Override
    default VERTEX matrixDeterminant() {
        return wrap(new MatrixDeterminantVertex<>(this));
    }

    @Override
    default IntegerVertex nanArgMax(int axis) {
        return new NaNArgMaxVertex<>(this, axis);
    }

    @Override
    default IntegerVertex nanArgMax() {
        return new NaNArgMaxVertex<>(this);
    }

    @Override
    default IntegerVertex nanArgMin(int axis) {
        return new NaNArgMinVertex<>(this, axis);
    }

    @Override
    default IntegerVertex nanArgMin() {
        return new NaNArgMinVertex<>(this);
    }
}
