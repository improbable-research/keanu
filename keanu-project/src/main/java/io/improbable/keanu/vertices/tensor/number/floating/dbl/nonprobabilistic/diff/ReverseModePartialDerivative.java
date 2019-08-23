package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;

public class ReverseModePartialDerivative {

    public static final ReverseModePartialDerivative EMPTY = new ReverseModePartialDerivative(null, null);

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

        if (this.isPresent() && addition.isPresent()) {
            return new ReverseModePartialDerivative(ofShape, partial.plus(addition.partial));
        } else if (this.isPresent() && !addition.isPresent()) {
            return new ReverseModePartialDerivative(ofShape, partial);
        } else if (!this.isPresent() && addition.isPresent()) {
            return new ReverseModePartialDerivative(addition.ofShape, addition.partial);
        } else {
            return ReverseModePartialDerivative.EMPTY;
        }
    }

    public ReverseModePartialDerivative subtract(ReverseModePartialDerivative subtraction) {

        if (this.isPresent() && subtraction.isPresent()) {
            return new ReverseModePartialDerivative(ofShape, partial.minus(subtraction.partial));
        } else if (this.isPresent() && !subtraction.isPresent()) {
            return new ReverseModePartialDerivative(ofShape, partial);
        } else if (!this.isPresent() && subtraction.isPresent()) {
            return new ReverseModePartialDerivative(subtraction.ofShape, subtraction.partial.unaryMinus());
        } else {
            return ReverseModePartialDerivative.EMPTY;
        }
    }

    public ReverseModePartialDerivative multiply(double multiplier) {

        if (!isPresent()) {
            return this;
        }

        return new ReverseModePartialDerivative(ofShape, partial.times(multiplier));
    }

    public ReverseModePartialDerivative multiply(DoubleTensor multiplier) {

        if (!isPresent()) {
            return this;
        }

        DoubleTensor result = partial.times(multiplier);

        return new ReverseModePartialDerivative(ofShape, result);
    }

    public static ReverseModePartialDerivative matrixMultiply(ReverseModePartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (!partial.isPresent()) {
            return partial;
        }

        final DoubleTensor partialValue = partial.get();

        final DoubleTensor result;
        if (partialIsLeft) {
            result = partialValue.matrixMultiply(multiplier.transpose());
        } else {
            result = multiplier.transpose().matrixMultiply(partialValue);
        }

        return new ReverseModePartialDerivative(partial.ofShape, result);
    }
}
