package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.vertices.dbl.DoubleVertex;

import java.util.Arrays;

public class PartialDerivativeVertex {

    private static final PartialDerivativeVertex EMPTY = null;

    private final DoubleVertex partial;

    public PartialDerivativeVertex(DoubleVertex partial) {
        this.partial = partial;
    }

    public boolean isPresent() {
        return partial != null;
    }

    public DoubleVertex get() {
        return partial;
    }

    public long[] getOfShape(long[] wrtShape) {
        return Arrays.copyOfRange(partial.getShape(), 0, partial.getRank() - wrtShape.length);
    }

    public long[] getWrtShape(long[] ofShape) {
        return Arrays.copyOfRange(partial.getShape(), ofShape.length, partial.getRank());
    }

    public PartialDerivativeVertex add(PartialDerivativeVertex addition) {

        if (this.isPresent() && addition.isPresent()) {
            return new PartialDerivativeVertex(partial.plus(addition.partial));
        } else if (this.isPresent() && !addition.isPresent()) {
            return new PartialDerivativeVertex(get());
        } else if (!this.isPresent() && addition.isPresent()) {
            return new PartialDerivativeVertex(addition.partial);
        } else {
            return PartialDerivativeVertex.EMPTY;
        }
    }

    public PartialDerivativeVertex subtract(PartialDerivativeVertex subtraction) {

        if (this.isPresent() && subtraction.isPresent()) {
            return new PartialDerivativeVertex(partial.minus(subtraction.partial));
        } else if (this.isPresent() && !subtraction.isPresent()) {
            return new PartialDerivativeVertex(get());
        } else if (!this.isPresent() && subtraction.isPresent()) {
            return new PartialDerivativeVertex(subtraction.partial.unaryMinus());
        } else {
            return PartialDerivativeVertex.EMPTY;
        }
    }

    public PartialDerivativeVertex multiplyBy(double multiplier) {

        if (!isPresent()) {
            return this;
        }

        return new PartialDerivativeVertex(partial.times(multiplier));
    }

    /**
     * This method assumes the partial 'of' rank is the same as the multiplier. This is usually the case except
     * for some broadcast operations.
     *
     * @param multiplier the value to multiply by
     * @return a partial derivative with of dimensions multiplied
     */
    public PartialDerivativeVertex multiplyAlongOfDimensions(DoubleVertex multiplier) {
        return multiplyAlongOfDimensions(multiplier, multiplier.getRank());
    }

    /**
     * @param multiplier    the value to multiply by
     * @param partialOfRank the rank of the 'of' part of the partial. This is needed if it is different
     *                      from the rank of the multiplier. This happens for rank changing broadcast ops.
     * @return a partial derivative with of dimensions multiplied
     */
    public PartialDerivativeVertex multiplyAlongOfDimensions(DoubleVertex multiplier, int partialOfRank) {

        if (!isPresent()) {
            return this;
        }

        DoubleVertex multiplierAlignedAlongOf = alignAlongOf(multiplier, partial.getShape(), partialOfRank);
        DoubleVertex result = partial.times(multiplierAlignedAlongOf);

        return new PartialDerivativeVertex(result);
    }

    public PartialDerivativeVertex divideByAlongOfDimensions(DoubleVertex divisor) {
        return divideByAlongOfDimensions(divisor, divisor.getRank());
    }

    public PartialDerivativeVertex divideByAlongOfDimensions(DoubleVertex divisor, int partialOfRank) {

        if (!isPresent()) {
            return this;
        }

        DoubleVertex divisorAlignedAlongOf = alignAlongOf(divisor, partial.getShape(), partialOfRank);
        DoubleVertex result = partial.div(divisorAlignedAlongOf);

        return new PartialDerivativeVertex(result);
    }

    public PartialDerivativeVertex multiplyAlongWrtDimensions(DoubleVertex multiplier) {

        if (!isPresent()) {
            return this;
        }

        DoubleVertex multiplierAlignedAlongWrt = alignAlongWrt(multiplier, partial.getRank());
        DoubleVertex result = partial.times(multiplierAlignedAlongWrt);

        return new PartialDerivativeVertex(result);
    }

    public static PartialDerivativeVertex matrixMultiplyAlongOfDimensions(PartialDerivativeVertex partial,
                                                                          DoubleVertex multiplier,
                                                                          boolean partialIsLeft) {

        if (!partial.isPresent()) {
            return partial;
        }

        final DoubleVertex partialValue = partial.get();
        final int partialRank = partialValue.getRank();

        DoubleVertex result;
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

        return new PartialDerivativeVertex(result);
    }

    public static PartialDerivativeVertex matrixMultiplyAlongWrtDimensions(PartialDerivativeVertex partial, DoubleVertex multiplier, boolean partialIsLeft) {

        if (!partial.isPresent()) {
            return partial;
        }

        final DoubleVertex partialValue = partial.get();
        final int partialRank = partialValue.getRank();
        final int wrtRightDimension = partialRank - 1;

        DoubleVertex result;
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

        return new PartialDerivativeVertex(result);
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
    private static DoubleVertex alignAlongOf(DoubleVertex tensor, long[] partialShape, int partialOfRank) {

        final long[] alongOfShape = new long[partialShape.length];
        Arrays.fill(alongOfShape, 1L);

        int tensorRank = tensor.getRank();
        System.arraycopy(tensor.getShape(), 0, alongOfShape, partialOfRank - tensorRank, tensorRank);

        return tensor.reshape(alongOfShape);
    }

    private static DoubleVertex alignAlongWrt(DoubleVertex tensor, int partialRank) {
        final long[] alongWrtShape = TensorShape.shapeToDesiredRankByPrependingOnes(tensor.getShape(), partialRank);
        return tensor.reshape(alongWrtShape);
    }
}
