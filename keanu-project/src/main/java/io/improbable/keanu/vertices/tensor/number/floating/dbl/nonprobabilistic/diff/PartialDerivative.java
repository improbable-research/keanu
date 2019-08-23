package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;

public class PartialDerivative {

    public static final PartialDerivative EMPTY = new PartialDerivative(null, null);

    private final DoubleTensor partial;
    private final long[] ofShape;

    public PartialDerivative(long[] ofShape, DoubleTensor partial) {
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

    public PartialDerivative add(PartialDerivative addition) {

        if (this.isPresent() && addition.isPresent()) {
            return new PartialDerivative(ofShape, partial.plus(addition.partial));
        } else if (this.isPresent() && !addition.isPresent()) {
            return new PartialDerivative(ofShape, partial);
        } else if (!this.isPresent() && addition.isPresent()) {
            return new PartialDerivative(addition.ofShape, addition.partial);
        } else {
            return PartialDerivative.EMPTY;
        }
    }

    public PartialDerivative subtract(PartialDerivative subtraction) {

        if (this.isPresent() && subtraction.isPresent()) {
            return new PartialDerivative(ofShape, partial.minus(subtraction.partial));
        } else if (this.isPresent() && !subtraction.isPresent()) {
            return new PartialDerivative(ofShape, partial);
        } else if (!this.isPresent() && subtraction.isPresent()) {
            return new PartialDerivative(subtraction.ofShape, subtraction.partial.unaryMinus());
        } else {
            return PartialDerivative.EMPTY;
        }
    }

    public PartialDerivative multiply(double multiplier) {

        if (!isPresent()) {
            return this;
        }

        return new PartialDerivative(ofShape, partial.times(multiplier));
    }

    public PartialDerivative multiply(DoubleTensor multiplier) {

        if (!isPresent()) {
            return this;
        }

        DoubleTensor result = partial.times(multiplier);

        return new PartialDerivative(ofShape, result);
    }

    public static PartialDerivative matrixMultiply(PartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

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

        return new PartialDerivative(partial.ofShape, result);
    }
}
