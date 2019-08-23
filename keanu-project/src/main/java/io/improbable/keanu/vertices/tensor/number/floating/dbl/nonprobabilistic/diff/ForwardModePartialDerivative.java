package io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;

public class ForwardModePartialDerivative {

    private final long[] wrtShape;

    public static final ForwardModePartialDerivative EMPTY = new ForwardModePartialDerivative(null, null);

    private final DoubleTensor partial;

    public ForwardModePartialDerivative(long[] wrtShape, DoubleTensor partial) {
        if (partial != null && wrtShape == null) {
            throw new IllegalArgumentException("must provide wrt shape");
        }
        this.wrtShape = wrtShape;
        this.partial = partial;
    }

    public boolean isPresent() {
        return partial != null;
    }

    public DoubleTensor get() {
        return partial;
    }

    public DoubleTensor getOfWrt() {
        if (isPresent()) {
            return swapDimsAtPivotPoint(partial, wrtShape.length);
        } else {
            return null;
        }
    }

    private static DoubleTensor swapDimsAtPivotPoint(DoubleTensor tensor, int pivotPoint) {
        int rank = tensor.getRank();
        int[] rearrange = new int[rank];

        for (int i = 0; i < rank; i++) {
            rearrange[i] = (pivotPoint + i) % rank;
        }

        return tensor.permute(rearrange);
    }

    public long[] getOfShape() {
        return Arrays.copyOfRange(partial.getShape(), wrtShape.length, partial.getRank());
    }

    public long[] getWrtShape() {
        return wrtShape;
    }

    public ForwardModePartialDerivative add(ForwardModePartialDerivative addition) {

        if (this.isPresent() && addition.isPresent()) {
            return new ForwardModePartialDerivative(wrtShape, partial.plus(addition.partial));
        } else if (this.isPresent() && !addition.isPresent()) {
            return new ForwardModePartialDerivative(wrtShape, get());
        } else if (!this.isPresent() && addition.isPresent()) {
            return new ForwardModePartialDerivative(addition.getWrtShape(), addition.partial);
        } else {
            return ForwardModePartialDerivative.EMPTY;
        }
    }

    public ForwardModePartialDerivative subtract(ForwardModePartialDerivative subtraction) {

        if (this.isPresent() && subtraction.isPresent()) {
            return new ForwardModePartialDerivative(wrtShape, partial.minus(subtraction.partial));
        } else if (this.isPresent() && !subtraction.isPresent()) {
            return new ForwardModePartialDerivative(wrtShape, get());
        } else if (!this.isPresent() && subtraction.isPresent()) {
            return new ForwardModePartialDerivative(subtraction.getWrtShape(), subtraction.partial.unaryMinus());
        } else {
            return ForwardModePartialDerivative.EMPTY;
        }
    }

    public ForwardModePartialDerivative multiplyBy(double multiplier) {

        if (!isPresent()) {
            return this;
        }

        return new ForwardModePartialDerivative(wrtShape, partial.times(multiplier));
    }

    public ForwardModePartialDerivative divideBy(DoubleTensor divisor) {

        if (!isPresent()) {
            return this;
        }

        DoubleTensor result = partial.div(divisor);

        return new ForwardModePartialDerivative(wrtShape, result);
    }

    public ForwardModePartialDerivative multiply(DoubleTensor multiplier) {

        if (!isPresent()) {
            return this;
        }

        DoubleTensor result = partial.times(multiplier);

        return new ForwardModePartialDerivative(wrtShape, result);
    }

    public static ForwardModePartialDerivative matrixMultiply(ForwardModePartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (!partial.isPresent()) {
            return partial;
        }

        final DoubleTensor partialValue;
        int ofRank = partial.get().getRank() - partial.getWrtShape().length;

        //Check for batch broadcasting
        if (ofRank < multiplier.getRank()) {
            long[] newShape = TensorShape.concat(partial.wrtShape, TensorShape.shapeToDesiredRankByPrependingOnes(partial.getOfShape(), multiplier.getRank()));
            partialValue = partial.get().reshape(newShape);
        } else {
            partialValue = partial.get();
        }

        final DoubleTensor result;
        if (partialIsLeft) {
            result = partialValue.matrixMultiply(multiplier);
        } else {
            result = multiplier.matrixMultiply(partialValue);
        }

        return new ForwardModePartialDerivative(partial.wrtShape, result);
    }
}
