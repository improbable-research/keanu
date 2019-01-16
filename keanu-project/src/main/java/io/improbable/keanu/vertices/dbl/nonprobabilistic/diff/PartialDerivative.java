package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
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

    public PartialDerivative multiplyBy(double multiplier) {

        if (!isPresent()) {
            return this;
        }

        return new PartialDerivative(partial.times(multiplier));
    }

    /**
     * This method assumes the partial 'of' rank is the same as the multiplier. This is usually the case except
     * for some broadcast operations.
     *
     * @param multiplier the value to multiply by
     * @return a partial derivative with of dimensions multiplied
     */
    public PartialDerivative multiplyAlongOfDimensions(DoubleTensor multiplier) {
        return multiplyAlongOfDimensions(multiplier, multiplier.getRank());
    }

    /**
     * @param multiplier    the value to multiply by
     * @param partialOfRank the rank of the 'of' part of the partial. This is needed if it is different
     *                      from the rank of the multiplier. This happens for rank changing broadcast ops.
     * @return a partial derivative with of dimensions multiplied
     */
    public PartialDerivative multiplyAlongOfDimensions(DoubleTensor multiplier, int partialOfRank) {

        if (!isPresent()) {
            return this;
        }

        DoubleTensor multiplierAlignedAlongOf = alignAlongOf(multiplier, partial.getShape(), partialOfRank);
        DoubleTensor result = partial.times(multiplierAlignedAlongOf);

        return new PartialDerivative(result);
    }

    public PartialDerivative divideByAlongOfDimensions(DoubleTensor divisor) {
        return divideByAlongOfDimensions(divisor, divisor.getRank());
    }

    public PartialDerivative divideByAlongOfDimensions(DoubleTensor divisor, int partialOfRank) {

        if (!isPresent()) {
            return this;
        }

        DoubleTensor divisorAlignedAlongOf = alignAlongOf(divisor, partial.getShape(), partialOfRank);
        DoubleTensor result = partial.div(divisorAlignedAlongOf);

        return new PartialDerivative(result);
    }

    public PartialDerivative multiplyAlongWrtDimensions(DoubleTensor multiplier) {

        if (!isPresent()) {
            return this;
        }

        DoubleTensor multiplierAlignedAlongWrt = alignAlongWrt(multiplier, partial.getRank());
        DoubleTensor result = partial.times(multiplierAlignedAlongWrt);

        return new PartialDerivative(result);
    }

    public PartialDerivative permute(int[] rearrange) {
        DoubleTensor result = partial.permute(rearrange);
        return new PartialDerivative(result);
    }

    public static PartialDerivative matrixMultiplyAlongOfDimensions(PartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (!partial.isPresent()) {
            return partial;
        }

        final DoubleTensor partialValue = partial.get();
        final int partialRank = partialValue.getRank();

        DoubleTensor result;
        if (partialIsLeft) {
            final int[] rearrange = TensorShape.dimensionRange(-1, partialRank - 1);
            rearrange[0] = 0;
            rearrange[1] = partialRank - 1;
            result = partialValue
                .tensorMultiply(multiplier, new int[]{1}, new int[]{0})
                .permute(rearrange);

        } else {
            result = multiplier
                .tensorMultiply(partialValue, new int[]{1}, new int[]{0});
        }

        return new PartialDerivative(result);
    }

    public static PartialDerivative matrixMultiplyAlongWrtDimensions(PartialDerivative partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (!partial.isPresent()) {
            return partial;
        }

        final DoubleTensor partialValue = partial.get();
        final int partialRank = partialValue.getRank();
        final int wrtRightDimension = partialRank - 1;

        DoubleTensor result;
        if (partialIsLeft) {
            result = partialValue
                .tensorMultiply(multiplier, new int[]{wrtRightDimension}, new int[]{1});
        } else {
            int wrtLeftDimension = partialRank - 2;
            int[] transposeWrt = TensorShape.dimensionRange(0, partialRank);
            transposeWrt[wrtRightDimension] = wrtLeftDimension;
            transposeWrt[wrtLeftDimension] = wrtRightDimension;

            result = partialValue
                .tensorMultiply(multiplier, new int[]{wrtLeftDimension}, new int[]{0})
                .permute(transposeWrt);
        }

        return new PartialDerivative(result);
    }

    /**
     * This is important for the case where the partial 'of' and the tensor are different ranks but are
     * still broadcastable.
     *
     * @param tensor        the tensor to align along the of dimensions
     * @param partialShape  the full shape of the partial in the format [of,wrt]
     * @param partialOfRank the rank of the 'of' part of the partial
     * @return a reshaped tensor with a shape of ones everywhere except the aligned of dimensions.
     * E.g.
     * tensorShape = [3,4]
     * partialShape = [2,3,4,5,6,7]
     * partialOfRank = 3
     * <p>
     * returned tensor will be of shape [1,3,4,1,1,1]
     */
    private static DoubleTensor alignAlongOf(DoubleTensor tensor, long[] partialShape, int partialOfRank) {

        final long[] alongOfShape = new long[partialShape.length];
        Arrays.fill(alongOfShape, 1L);

        int tensorRank = tensor.getRank();
        System.arraycopy(tensor.getShape(), 0, alongOfShape, partialOfRank - tensorRank, tensorRank);

        return tensor.reshape(alongOfShape);
    }

    private static DoubleTensor alignAlongWrt(DoubleTensor tensor, int partialRank) {
        final long[] alongWrtShape = TensorShape.shapeToDesiredRankByPrependingOnes(tensor.getShape(), partialRank);
        return tensor.reshape(alongWrtShape);
    }
}
