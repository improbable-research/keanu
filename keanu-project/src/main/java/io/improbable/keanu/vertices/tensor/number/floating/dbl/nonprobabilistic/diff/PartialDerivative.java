package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;

public class PartialDerivative {

    public static final PartialDerivative EMPTY = new PartialDerivative(null);

    private final DoubleTensor partial;

    public PartialDerivative(DoubleTensor partial) {
        this.partial = partial;
    }

    public boolean isPresent() {
        return partial != null;
    }

    public DoubleTensor get() {
        return partial;
    }

    public long[] getOfShape(long[] wrtShape) {
        return Arrays.copyOfRange(partial.getShape(), 0, partial.getRank() - wrtShape.length);
    }

    public long[] getWrtShape(long[] ofShape) {
        return Arrays.copyOfRange(partial.getShape(), ofShape.length, partial.getRank());
    }

    public PartialDerivative add(PartialDerivative addition) {

        if (this.isPresent() && addition.isPresent()) {
            return new PartialDerivative(partial.plus(addition.partial));
        } else if (this.isPresent() && !addition.isPresent()) {
            return new PartialDerivative(get());
        } else if (!this.isPresent() && addition.isPresent()) {
            return new PartialDerivative(addition.partial);
        } else {
            return PartialDerivative.EMPTY;
        }
    }

    public PartialDerivative subtract(PartialDerivative subtraction) {

        if (this.isPresent() && subtraction.isPresent()) {
            return new PartialDerivative(partial.minus(subtraction.partial));
        } else if (this.isPresent() && !subtraction.isPresent()) {
            return new PartialDerivative(get());
        } else if (!this.isPresent() && subtraction.isPresent()) {
            return new PartialDerivative(subtraction.partial.unaryMinus());
        } else {
            return PartialDerivative.EMPTY;
        }
    }

    public PartialDerivative multiply(double multiplier) {

        if (!isPresent()) {
            return this;
        }

        return new PartialDerivative(partial.times(multiplier));
    }

    public PartialDerivative multiply(DoubleTensor multiplier) {

        if (!isPresent()) {
            return this;
        }

        DoubleTensor result = partial.times(multiplier);

        return new PartialDerivative(result);
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

        return new PartialDerivative(result);
    }
}
