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

    public ForwardModePartialDerivative add(ForwardModePartialDerivative right, long[] desiredOfShape) {

        if (this.isPresent() && right.isPresent()) {

            if (partial.getRank() == right.partial.getRank()) {
                return new ForwardModePartialDerivative(wrtShape, partial.plus(right.partial));
            } else if (partial.getRank() < right.partial.getRank()) {
                final DoubleTensor leftPartial = upRankForBroadcast(this, desiredOfShape.length);
                return new ForwardModePartialDerivative(wrtShape, leftPartial.plus(right.partial));
            } else {
                final DoubleTensor rightPartial = upRankForBroadcast(right, desiredOfShape.length);
                return new ForwardModePartialDerivative(wrtShape, partial.plus(rightPartial));
            }
        } else if (this.isPresent() && !right.isPresent()) {
            final DoubleTensor resultValue = broadcastToDesiredShape(this, desiredOfShape);
            return new ForwardModePartialDerivative(wrtShape, resultValue);
        } else if (!this.isPresent() && right.isPresent()) {
            final DoubleTensor resultValue = broadcastToDesiredShape(right, desiredOfShape);
            return new ForwardModePartialDerivative(right.wrtShape, resultValue);
        } else {
            return ForwardModePartialDerivative.EMPTY;
        }
    }

    public ForwardModePartialDerivative subtract(ForwardModePartialDerivative right, long[] desiredOfShape) {

        if (this.isPresent() && right.isPresent()) {

            if (partial.getRank() == right.partial.getRank()) {
                return new ForwardModePartialDerivative(wrtShape, partial.minus(right.partial));
            } else if (partial.getRank() < right.partial.getRank()) {
                final DoubleTensor leftPartial = upRankForBroadcast(this, desiredOfShape.length);
                return new ForwardModePartialDerivative(wrtShape, leftPartial.minus(right.partial));
            } else {
                final DoubleTensor rightPartial = upRankForBroadcast(right, desiredOfShape.length);
                return new ForwardModePartialDerivative(wrtShape, partial.minus(rightPartial));
            }
        } else if (this.isPresent() && !right.isPresent()) {
            final DoubleTensor resultValue =  broadcastToDesiredShape(this, desiredOfShape);
            return new ForwardModePartialDerivative(wrtShape, resultValue);
        } else if (!this.isPresent() && right.isPresent()) {
            final DoubleTensor resultValue =  broadcastToDesiredShape(right.unaryMinus(), desiredOfShape);
            return new ForwardModePartialDerivative(right.wrtShape, resultValue);
        } else {
            return ForwardModePartialDerivative.EMPTY;
        }
    }

    private DoubleTensor broadcastToDesiredShape(ForwardModePartialDerivative fromPartial, long[] desiredOfShape) {
        final long[] resultShape = TensorShape.concat(fromPartial.wrtShape, desiredOfShape);
        DoubleTensor partialValue = upRankForBroadcast(fromPartial, desiredOfShape.length);

        if (Arrays.equals(partialValue.getShape(), resultShape)) {
            return partialValue;
        } else {
            return partialValue.broadcast(resultShape);
        }
    }

    private static DoubleTensor upRankForBroadcast(ForwardModePartialDerivative partial, int desiredOfRank) {
        int partialOfRank = partial.get().getRank() - partial.wrtShape.length;

        if (partialOfRank < desiredOfRank) {
            long[] newShape = TensorShape.concat(partial.wrtShape, TensorShape.shapeToDesiredRankByPrependingOnes(partial.getOfShape(), desiredOfRank));
            return partial.get().reshape(newShape);
        } else {
            return partial.get();
        }
    }

    public ForwardModePartialDerivative divideBy(DoubleTensor divisor) {

        if (!isPresent()) {
            return this;
        }

        final DoubleTensor partialValue = upRankForBroadcast(this, divisor.getRank());
        DoubleTensor result = partialValue.div(divisor);

        return new ForwardModePartialDerivative(wrtShape, result);
    }

    public ForwardModePartialDerivative unaryMinus() {

        if (!isPresent()) {
            return this;
        }

        return new ForwardModePartialDerivative(wrtShape, partial.unaryMinus());
    }

    public ForwardModePartialDerivative multiply(DoubleTensor multiplier) {

        if (!isPresent()) {
            return this;
        }

        final DoubleTensor partialValue = upRankForBroadcast(this, multiplier.getRank());
        DoubleTensor result = partialValue.times(multiplier);

        return new ForwardModePartialDerivative(wrtShape, result);
    }

    public static ForwardModePartialDerivative matrixMultiply(ForwardModePartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (!partial.isPresent()) {
            return partial;
        }

        final DoubleTensor partialValue = upRankForBroadcast(partial, multiplier.getRank());

        final DoubleTensor result;
        if (partialIsLeft) {
            result = partialValue.matrixMultiply(multiplier);
        } else {
            result = multiplier.matrixMultiply(partialValue);
        }

        return new ForwardModePartialDerivative(partial.wrtShape, result);
    }

}
