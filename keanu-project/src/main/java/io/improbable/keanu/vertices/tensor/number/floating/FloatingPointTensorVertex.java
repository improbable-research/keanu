package io.improbable.keanu.vertices.tensor.number.floating;

import io.improbable.keanu.BaseFloatingPointTensor;
import io.improbable.keanu.tensor.FloatingPointTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.NaNArgMaxVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.NaNArgMinVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcCosVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcSinVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ArcTanVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CeilVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.CosVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.ExpVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.FloorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogGammaVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.LogVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MatrixDeterminantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.MatrixInverseVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.RoundVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SigmoidVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.SinVertex;
import io.improbable.keanu.vertices.tensor.number.floating.operators.unary.TanVertex;

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
    default VERTEX average() {
        return null;
    }

    @Override
    default VERTEX standardDeviation() {
        return null;
    }

    @Override
    default VERTEX standardize() {
        return null;
    }

    @Override
    default VERTEX logAddExp2(VERTEX that) {
        return null;
    }

    @Override
    default VERTEX logAddExp(VERTEX that) {
        return null;
    }

    @Override
    default VERTEX log1p() {
        return null;
    }

    @Override
    default VERTEX log2() {
        return null;
    }

    @Override
    default VERTEX log10() {
        return null;
    }

    @Override
    default VERTEX exp2() {
        return null;
    }

    @Override
    default VERTEX expM1() {
        return null;
    }

    @Override
    default VERTEX replaceNaN(T value) {
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
        return null;
    }

    @Override
    default VERTEX sigmoid() {
        return wrap(new SigmoidVertex<>(this));
    }

    @Override
    default VERTEX choleskyDecomposition() {
        return null;
    }

    @Override
    default VERTEX sinh() {
        return null;
    }

    @Override
    default VERTEX cosh() {
        return null;
    }

    @Override
    default VERTEX tanh() {
        return null;
    }

    @Override
    default VERTEX asinh() {
        return null;
    }

    @Override
    default VERTEX acosh() {
        return null;
    }

    @Override
    default VERTEX atanh() {
        return null;
    }

    @Override
    default VERTEX atan() {
        return wrap(new ArcTanVertex<>(this));
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
