package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;

public class ReverseModePartialDerivative {

    private final DoubleTensor partial;
    private final long[] ofShape;

    public ReverseModePartialDerivative(long[] ofShape, DoubleTensor partial) {
        if (partial != null && ofShape == null) {
            throw new IllegalArgumentException("Must provide of shape");
        }
        this.ofShape = ofShape;
        this.partial = partial;
    }

    public boolean isPresent() {
        return partial != null;
    }

    public DoubleTensor get() {
        return partial;
    }

    public long[] getOfShape() {
        return ofShape;
    }

    public long[] getWrtShape() {
        return Arrays.copyOfRange(partial.getShape(), ofShape.length, partial.getRank());
    }

    public ReverseModePartialDerivative add(ReverseModePartialDerivative addition) {
        return new ReverseModePartialDerivative(ofShape, partial.plus(addition.partial));
    }

    public ReverseModePartialDerivative multiply(double multiplier) {
        return new ReverseModePartialDerivative(ofShape, partial.times(multiplier));
    }

    public ReverseModePartialDerivative multiply(DoubleTensor multiplier) {

        DoubleTensor result = partial.times(multiplier);

        return new ReverseModePartialDerivative(ofShape, result);
    }

    public static ReverseModePartialDerivative matrixMultiply(ReverseModePartialDerivative partial,
                                                              DoubleTensor multiplier,
                                                              boolean partialIsLeft,
                                                              boolean transposePartial,
                                                              boolean transposeMultiplier) {

        final DoubleTensor partialValue = partial.get();

        final DoubleTensor result;
        if (partialIsLeft) {
            result = partialValue.matrixMultiply(multiplier, transposePartial, transposeMultiplier);
        } else {
            result = multiplier.matrixMultiply(partialValue, transposeMultiplier, transposePartial);
        }

        return new ReverseModePartialDerivative(partial.ofShape, result);
    }
}
