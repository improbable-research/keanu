package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Arrays;

public class PartialDerivative {

    public static final PartialDerivative EMPTY = new PartialDerivative();

    private final DoubleTensor partial;

    public PartialDerivative(DoubleTensor partial) {
        this.partial = partial;
    }

    private PartialDerivative() {
        this.partial = null;
    }

    public boolean isPresent() {
        return partial != null;
    }

    public boolean isEmpty() {
        return !isPresent();
    }

    public DoubleTensor get() {
        return partial;
    }

    public long[] getOfShape(long[] wrtShape) {
        return Arrays.copyOfRange(partial.getShape(), 0, partial.getShape().length - wrtShape.length);
    }

    public long[] getWrtShape(long[] ofShape) {
        return Arrays.copyOfRange(partial.getShape(), ofShape.length, partial.getShape().length);
    }

    public PartialDerivative add(PartialDerivative addition) {

        if (isPresent() && addition.isPresent()) {
            return new PartialDerivative(partial.plus(addition.partial));
        } else if (isPresent() && addition.isEmpty()) {
            return new PartialDerivative(get());
        } else if (isEmpty() && addition.isPresent()) {
            return new PartialDerivative(addition.partial);
        } else {
            return PartialDerivative.EMPTY;
        }
    }

    public PartialDerivative subtract(PartialDerivative subtraction) {

        if (isPresent() && subtraction.isPresent()) {
            return new PartialDerivative(partial.minus(subtraction.partial));
        } else if (isPresent() && subtraction.isEmpty()) {
            return new PartialDerivative(get());
        } else if (isEmpty() && subtraction.isPresent()) {
            return new PartialDerivative(subtraction.partial.unaryMinus());
        } else {
            return PartialDerivative.EMPTY;
        }
    }

    public PartialDerivative multiplyBy(double multiplier) {

        if (isEmpty()) {
            return this;
        }

        return new PartialDerivative(partial.times(multiplier));
    }

    public PartialDerivative multiplyAlongOfDimensions(DoubleTensor multiplier) {

        if (isEmpty()) {
            return this;
        }

        DoubleTensor multiplierFromLeft = increaseRankByAppendingOnesToShape(multiplier, partial.getRank());
        DoubleTensor result = partial.times(multiplierFromLeft);

        return new PartialDerivative(result);
    }

    public PartialDerivative multiplyAlongWrtDimensions(DoubleTensor multiplier) {

        if (isEmpty()) {
            return this;
        }

        DoubleTensor multiplierFromRight = increaseRankByPrependingOnesToShape(multiplier, partial.getRank());
        DoubleTensor result = partial.times(multiplierFromRight);

        return new PartialDerivative(result);
    }

    public PartialDerivative divideByAlongOfDimensions(DoubleTensor divisor) {

        if (isEmpty()) {
            return this;
        }

        DoubleTensor divisorFromLeft = increaseRankByAppendingOnesToShape(divisor, partial.getRank());
        DoubleTensor result = partial.div(divisorFromLeft);

        return new PartialDerivative(result);
    }

    public static PartialDerivative matrixMultiplyAlongOfDimensions(PartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (partial.isEmpty()) {
            return partial;
        }

        int partialRank = partial.get().getRank();

        DoubleTensor result;
        if (partialIsLeft) {
            int[] rearrange = TensorShape.dimensionRange(-1, partialRank - 1);
            rearrange[0] = 0;
            rearrange[1] = partialRank - 1;
            result = partial.get()
                .tensorMultiply(multiplier, new int[]{1}, new int[]{0})
                .permute(rearrange);

        } else {
            result = multiplier
                .tensorMultiply(partial.get(), new int[]{1}, new int[]{0});
        }

        return new PartialDerivative(result);
    }

    public static PartialDerivative matrixMultiplyAlongWrtDimensions(PartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (partial.isEmpty()) {
            return partial;
        }

        int partialRank = partial.get().getRank();

        int wrtRightDimension = partialRank - 1;
        int wrtLeftDimension = partialRank - 2;

        DoubleTensor result;
        if (partialIsLeft) {
            result = partial.get()
                .tensorMultiply(multiplier, new int[]{wrtRightDimension}, new int[]{1});
        } else {
            int[] transposeWrt = TensorShape.dimensionRange(0, partialRank);
            transposeWrt[wrtRightDimension] = wrtLeftDimension;
            transposeWrt[wrtLeftDimension] = wrtRightDimension;

            result = partial.get()
                .tensorMultiply(multiplier, new int[]{wrtLeftDimension}, new int[]{0})
                .permute(transposeWrt);
        }

        return new PartialDerivative(result);
    }

    public static DoubleTensor increaseRankByAppendingOnesToShape(DoubleTensor lowRankTensor, int desiredRank) {
        return lowRankTensor.reshape(
            TensorShape.shapeDesiredToRankByAppendingOnes(lowRankTensor.getShape(), desiredRank)
        );
    }

    public static DoubleTensor increaseRankByPrependingOnesToShape(DoubleTensor lowRankTensor, int desiredRank) {
        return lowRankTensor.reshape(
            TensorShape.shapeToDesiredRankByPrependingOnes(lowRankTensor.getShape(), desiredRank)
        );
    }
}
