package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;

import java.util.Arrays;

public class PartialDerivative {

    public static final PartialDerivative EMPTY = new PartialDerivative();

    private final VertexId id;
    private final DoubleTensor partial;

    public PartialDerivative(VertexId id, DoubleTensor partial) {
        this.id = id;
        this.partial = partial;
    }

    private PartialDerivative() {
        this.id = null;
        this.partial = null;
    }

    public boolean isPresent() {
        return partial != null;
    }

    public boolean isEmpty() {
        return !isPresent();
    }

    public DoubleTensor getPartial() {
        return partial;
    }

    public VertexId getKey() {
        return id;
    }

    public long[] getOfShape(long[] wrtShape) {
        return Arrays.copyOfRange(partial.getShape(), 0, partial.getShape().length - wrtShape.length);
    }

    public long[] getWrtShape(long[] ofShape) {
        return Arrays.copyOfRange(partial.getShape(), ofShape.length, partial.getShape().length);
    }

    /**
     * This will sum partial derivatives that are represented as tensors over given dimensions.
     * The dimensions that are summed over will be reshaped to the specified resultShape.
     *
     * @param dimensions  dimensions to sum over
     * @return summed and reshaped partials
     */
    public PartialDerivative sumOverOfDimensions(int[] dimensions) {

        return new PartialDerivative(getKey(), partial.sum(dimensions));
    }

    /**
     * This will sum partial derivatives that are represented as tensors over given dimensions.
     * The dimensions that are summed over will be reshaped to the specified resultShape.
     *
     * @param dimensions  dimensions to sum over
     * @param resultShape shape of sum result
     * @return summed and reshaped partials
     */
    public PartialDerivative sumOverWrtDimensions(int[] dimensions, long[] resultShape) {

        if (isEmpty()) {
            return this;
        }

        if (dimensions.length == 0) {
            return new PartialDerivative(getKey(), getPartial());
        }

        DoubleTensor v = getPartial();
        long[] vShape = v.getShape();
        long[] ofShape = TensorShape.selectDimensions(0, vShape.length - dimensions.length, vShape);

        DoubleTensor summedV = v.sum(dimensions);
        long[] newShape = TensorShape.concat(ofShape, resultShape);
        summedV = summedV.reshape(newShape);

        return new PartialDerivative(getKey(), summedV);
    }

    public PartialDerivative add(PartialDerivative addition) {

        if (isPresent() && addition.isPresent()) {
            return new PartialDerivative(getKey(), partial.plus(addition.partial));
        } else if (isPresent() && addition.isEmpty()) {
            return new PartialDerivative(getKey(), getPartial());
        } else if (isEmpty() && addition.isPresent()) {
            return new PartialDerivative(addition.getKey(), addition.partial);
        } else {
            return PartialDerivative.EMPTY;
        }
    }

    public PartialDerivative subtract(PartialDerivative subtraction) {

        if (isPresent() && subtraction.isPresent()) {
            return new PartialDerivative(getKey(), partial.minus(subtraction.partial));
        } else if (isPresent() && subtraction.isEmpty()) {
            return new PartialDerivative(getKey(), getPartial());
        } else if (isEmpty() && subtraction.isPresent()) {
            return new PartialDerivative(subtraction.getKey(), subtraction.partial.unaryMinus());
        } else {
            return PartialDerivative.EMPTY;
        }
    }

    public PartialDerivative multiplyBy(double multiplier) {

        if (isEmpty()) {
            return this;
        }

        return new PartialDerivative(id, partial.times(multiplier));
    }

    public PartialDerivative multiplyAlongOfDimensions(DoubleTensor multiplier) {

        if (isEmpty()) {
            return this;
        }

        DoubleTensor multiplierFromLeft = increaseRankByAppendingOnesToShape(multiplier, partial.getRank());
        DoubleTensor result = partial.times(multiplierFromLeft);

        return new PartialDerivative(id, result);
    }

    public PartialDerivative multiplyAlongWrtDimensions(DoubleTensor multiplier) {

        if (isEmpty()) {
            return this;
        }

        DoubleTensor multiplierFromRight = increaseRankByPrependingOnesToShape(multiplier, partial.getRank());
        DoubleTensor result = partial.times(multiplierFromRight);

        return new PartialDerivative(id, result);
    }

    public PartialDerivative divideByAlongOfDimensions(DoubleTensor divisor) {

        if (isEmpty()) {
            return this;
        }

        DoubleTensor divisorFromLeft = increaseRankByAppendingOnesToShape(divisor, partial.getRank());
        DoubleTensor result = partial.div(divisorFromLeft);

        return new PartialDerivative(id, result);
    }

    public static PartialDerivative matrixMultiplyAlongOfDimensions(PartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (partial.isEmpty()) {
            return partial;
        }

        int partialRank = partial.getPartial().getRank();

        DoubleTensor result;
        if (partialIsLeft) {
            int[] rearrange = TensorShape.dimensionRange(-1, partialRank - 1);
            rearrange[0] = 0;
            rearrange[1] = partialRank - 1;
            result = partial.getPartial()
                .tensorMultiply(multiplier, new int[]{1}, new int[]{0})
                .permute(rearrange);

        } else {
            result = multiplier
                .tensorMultiply(partial.getPartial(), new int[]{1}, new int[]{0});
        }

        return new PartialDerivative(partial.getKey(), result);
    }

    public static PartialDerivative matrixMultiplyAlongWrtDimensions(PartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (partial.isEmpty()) {
            return partial;
        }

        int partialRank = partial.getPartial().getRank();

        int wrtRightDimension = partialRank - 1;
        int wrtLeftDimension = partialRank - 2;

        DoubleTensor result;
        if (partialIsLeft) {
            result = partial.getPartial()
                .tensorMultiply(multiplier, new int[]{wrtRightDimension}, new int[]{1});
        } else {
            int[] transposeWrt = TensorShape.dimensionRange(0, partialRank);
            transposeWrt[wrtRightDimension] = wrtLeftDimension;
            transposeWrt[wrtLeftDimension] = wrtRightDimension;

            result = partial.getPartial()
                .tensorMultiply(multiplier, new int[]{wrtLeftDimension}, new int[]{0})
                .permute(transposeWrt);
        }

        return new PartialDerivative(partial.getKey(), result);
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
