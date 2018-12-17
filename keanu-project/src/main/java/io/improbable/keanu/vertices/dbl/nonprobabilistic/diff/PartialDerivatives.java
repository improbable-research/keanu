package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.VertexId;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;

public class PartialDerivatives {

    public static final PartialDerivatives OF_CONSTANT = new PartialDerivatives(null, null);

    public static PartialDerivatives withRespectToSelf(VertexId withRespectTo, long[] shape) {
        return new PartialDerivatives(
            withRespectTo,
            DoubleTensor.eye((int) TensorShape.getLength(shape)).reshape(TensorShape.concat(shape, shape))
        );
    }

    private final VertexId id;
    private final DoubleTensor value;

    public PartialDerivatives(VertexId id, DoubleTensor derivativeWithRespectTo) {
        this.id = id;
        this.value = derivativeWithRespectTo;
    }

    public PartialDerivatives() {
        this.id = null;
        this.value = null;
    }

    public boolean isEmpty() {
        return value == null;
    }

    public DoubleTensor getValue() {
        return value;
    }

    public VertexId getKey() {
        return id;
    }

    public boolean isKey(VertexId id) {
        VertexId key = getKey();

        if (key == null) {
            return id == null;
        }

        return key.equals(id);
    }

    /**
     * This will sum partial derivatives that are represented as tensors over given dimensions.
     * The dimensions that are summed over will be reshaped to the specified resultShape.
     *
     * @param dimensions  dimensions to sum over
     * @param resultShape shape of sum result
     * @param ofRank      the rank of the "of" part of the partials
     * @return summed and reshaped partials
     */
    public PartialDerivatives sumOverOfDimensions(int[] dimensions, long[] resultShape, int ofRank) {

        DoubleTensor v = getValue();
        long[] vShape = v.getShape();
        long[] wrtShape = TensorShape.selectDimensions(ofRank, vShape.length, vShape);

        DoubleTensor summedV = v.sum(dimensions);
        long[] newShape = TensorShape.concat(resultShape, wrtShape);
        summedV = summedV.reshape(newShape);

        return new PartialDerivatives(getKey(), summedV);
    }

    /**
     * This will sum partial derivatives that are represented as tensors over given dimensions.
     * The dimensions that are summed over will be reshaped to the specified resultShape.
     *
     * @param dimensions  dimensions to sum over
     * @param resultShape shape of sum result
     * @param wrtRank     the rank of the "wrt" part of the partials
     * @return summed and reshaped partials
     */
    public PartialDerivatives sumOverWrtDimensions(int[] dimensions, long[] resultShape, int wrtRank) {

        if (dimensions.length == 0) {
            return new PartialDerivatives(getKey(), getValue());
        }

        DoubleTensor v = getValue();
        long[] vShape = v.getShape();
        long[] ofShape = TensorShape.selectDimensions(0, v.getShape().length - wrtRank, vShape);

        DoubleTensor summedV = v.sum(dimensions);
        long[] newShape = TensorShape.concat(ofShape, resultShape);
        summedV = summedV.reshape(newShape);

        return new PartialDerivatives(getKey(), summedV);
    }

    public PartialDerivatives add(PartialDerivatives toAdd) {
        return add(toAdd, false, false, null);
    }

    public PartialDerivatives add(PartialDerivatives addition, boolean leftIsLengthOne, boolean rightIsLengthOne, long[] resultShape) {

        DoubleTensor added = cloneWithCorrectShape(value, leftIsLengthOne, resultShape);
        DoubleTensor toAdd = cloneWithCorrectShape(addition.getValue(), rightIsLengthOne, resultShape);

        if (added == null && toAdd != null) {
            return new PartialDerivatives(addition.getKey(), toAdd);
        } else if (added != null && toAdd == null) {
            return new PartialDerivatives(getKey(), added);
        } else if (added != null && toAdd != null) {
            return new PartialDerivatives(getKey(), added.plus(toAdd));
        } else {
            return new PartialDerivatives();
        }

    }

    public PartialDerivatives subtract(PartialDerivatives subtraction) {
        return subtract(subtraction, false, false, null);
    }

    public PartialDerivatives subtract(PartialDerivatives subtraction, boolean leftIsLengthOne, boolean rightIsLengthOne, long[] resultShape) {

        DoubleTensor subtracted = cloneWithCorrectShape(value, leftIsLengthOne, resultShape);
        DoubleTensor toSubtract = cloneWithCorrectShape(subtraction.getValue(), rightIsLengthOne, resultShape);

        if (subtracted == null && toSubtract != null) {
            return new PartialDerivatives(subtraction.getKey(), toSubtract.unaryMinus());
        } else if (subtracted != null && toSubtract == null) {
            return new PartialDerivatives(getKey(), subtracted);
        } else if (subtracted != null && toSubtract != null) {
            return new PartialDerivatives(getKey(), subtracted.minus(toSubtract));
        } else {
            return new PartialDerivatives();
        }

    }

    private static DoubleTensor cloneWithCorrectShape(DoubleTensor v,
                                                      boolean ofIsLengthOne,
                                                      long[] resultShape) {

        if (v == null) {
            //TODO: stop this from happening
            return null;
        }

        if (ofIsLengthOne) {
            return DoubleTensor.zeros(TensorShape.concat(resultShape, v.getShape())).plus(v);
        }

        return v;
    }

    public PartialDerivatives multiplyAlongOfDimensions(DoubleTensor multiplier, long[] ofShape) {

        if (isEmpty()) {
            return clone();
        }

        DoubleTensor result;

        if (multiplier.isScalar()) {
            result = value.times(multiplier.scalar());
        } else {
            result = elementWiseMultiplyAlongOf(value, multiplier, ofShape);
        }

        return new PartialDerivatives(id, result);
    }

    public PartialDerivatives multiplyAlongWrtDimensions(DoubleTensor multiplier, long[] wrtShape) {

        if (isEmpty()) {
            return clone();
        }

        DoubleTensor result;
        if (multiplier.isScalar()) {
            result = value.times(multiplier.scalar());
        } else {
            result = elementWiseMultiplyAlongWrt(value, multiplier, wrtShape);
        }

        return new PartialDerivatives(id, result);
    }

    private DoubleTensor elementWiseMultiplyAlongOf(DoubleTensor partial, DoubleTensor multiplier, long[] ofShape) {

        long[] partialOfShape = extractOfShape(partial.getShape(), ofShape.length);

        boolean needsBroadcast = !Arrays.equals(partialOfShape, multiplier.getShape());
        if (needsBroadcast) {
            long[] partialWrtShape = extractWrtShape(partial.getShape(), ofShape.length);
            long[] broadcastedOfShape = Shape.broadcastOutputShape(multiplier.getShape(), partialOfShape);
            long[] resultShape = TensorShape.concat(broadcastedOfShape, partialWrtShape);

            DoubleTensor multiplierFromLeft = increaseRankByAppendingOnesToShape(multiplier, resultShape.length);
            DoubleTensor appropriateShapePartial = increaseRankByPrependingOnesToShape(partial, resultShape.length);

            return DoubleTensor.ones(resultShape).times(appropriateShapePartial).times(multiplierFromLeft);
        }

        DoubleTensor multiplierFromLeft = increaseRankByAppendingOnesToShape(multiplier, partial.getRank());
        return partial.times(multiplierFromLeft);
    }

    private DoubleTensor elementWiseMultiplyAlongWrt(DoubleTensor partial, DoubleTensor multiplier, long[] wrtShape) {

        long[] partialWrtShape = extractWrtShape(partial.getShape(), partial.getRank() - wrtShape.length);

        boolean needsBroadcast = !Arrays.equals(partialWrtShape, multiplier.getShape());
        if (needsBroadcast) {

            long[] partialOfShape = extractOfShape(partial.getShape(), partial.getRank() - wrtShape.length);
            long[] broadcastedWrtShape = Shape.broadcastOutputShape(multiplier.getShape(), partialWrtShape);
            long[] resultShape = TensorShape.concat(partialOfShape, broadcastedWrtShape);

            DoubleTensor multiplierFromRight = increaseRankByPrependingOnesToShape(multiplier, resultShape.length);
            DoubleTensor appropriateShapePartial = increaseRankByAppendingOnesToShape(partial, resultShape.length);

            return DoubleTensor.ones(resultShape).times(appropriateShapePartial).times(multiplierFromRight);
        }

        DoubleTensor multiplierFromRight = increaseRankByPrependingOnesToShape(multiplier, partial.getRank());
        return partial.times(multiplierFromRight);
    }

    public static PartialDerivatives matrixMultiplyAlongOfDimensions(PartialDerivatives partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (partial.isEmpty()) {
            partial.clone();
        }

        int partialRank = partial.getValue().getRank();

        DoubleTensor result;
        if (partialIsLeft) {
            int[] rearrange = TensorShape.dimensionRange(-1, partialRank - 1);
            rearrange[0] = 0;
            rearrange[1] = partialRank - 1;
            result = partial.getValue()
                .tensorMultiply(multiplier, new int[]{1}, new int[]{0})
                .permute(rearrange);

        } else {
            result = multiplier
                .tensorMultiply(partial.getValue(), new int[]{1}, new int[]{0});
        }

        return new PartialDerivatives(partial.getKey(), result);
    }

    public static PartialDerivatives matrixMultiplyAlongWrtDimensions(PartialDerivatives partial, DoubleTensor multiplier, boolean partialIsLeft) {

        if (partial.isEmpty()) {
            return partial.clone();
        }

        int partialRank = partial.getValue().getRank();

        int wrtRightDimension = partialRank - 1;
        int wrtLeftDimension = partialRank - 2;

        DoubleTensor result;
        if (partialIsLeft) {
            result = partial.getValue()
                .tensorMultiply(multiplier, new int[]{wrtRightDimension}, new int[]{1});
        } else {
            int[] transposeWrt = TensorShape.dimensionRange(0, partialRank);
            transposeWrt[wrtRightDimension] = wrtLeftDimension;
            transposeWrt[wrtLeftDimension] = wrtRightDimension;

            result = partial.getValue()
                .tensorMultiply(multiplier, new int[]{wrtLeftDimension}, new int[]{0})
                .permute(transposeWrt);
        }

        return new PartialDerivatives(partial.getKey(), result);
    }

    public PartialDerivatives multiplyBy(double multiplier) {

        if (isEmpty()) {
            return clone();
        }

        return new PartialDerivatives(id, value.times(multiplier));
    }

    public PartialDerivatives divideBy(DoubleTensor divisor) {

        if (isEmpty()) {
            return clone();
        }

        DoubleTensor partial = getValue();
        DoubleTensor result = partial.div(increaseRankByAppendingOnesToShape(divisor, partial.getRank()));

        return new PartialDerivatives(id, result);
    }

    public PartialDerivatives clone() {
        return new PartialDerivatives(id, value);
    }

    public PartialDerivatives reshape(int currentRank, long[] proposedShape) {

        if (isEmpty()) {
            return clone();
        }

        long[] wrtShape = extractWrtShape(value.getShape(), currentRank);
        long[] newPartialShape = TensorShape.concat(proposedShape, wrtShape);

        return new PartialDerivatives(id, value.reshape(newPartialShape));
    }

    /**
     * Slice the partials along dimension at a specified index.
     *
     * @param dimension dimension to slice along
     * @param index     index to slice at
     * @param reshape   Due to the way our tensor implementation works, slicing a rank 2 tensor gives us a rank two back, whereas
     *                  slicing a higher rank tensor gives you a (rank - 1) tensor back.  This causes problems for rank 2 tensors
     *                  where the shape of the "of" will go from, say, 3x3 to 3x1 whereas the partial will go from 3x3x3x3 to
     *                  3x3x3 instead of 3x1x3x3.  This reshape deals with this case.  Only needed for rank two inputs as higher
     *                  ranks correctly resolve (eg 3x3x3 will have a 3x3x3x3x3x3 and after slicing will be a 3x3 and a partial
     *                  of 3x3x3x3x3.
     * @return the sliced partials
     */
    public PartialDerivatives slice(int dimension, long index, boolean reshape) {

        if (isEmpty()) {
            return clone();
        }

        long[] partialDerivativeShape = Arrays.copyOf(value.getShape(), value.getShape().length);
        partialDerivativeShape[dimension] = 1;
        DoubleTensor slicedPartialDerivative = value.slice(dimension, index);

        if (reshape) {
            slicedPartialDerivative = slicedPartialDerivative.reshape(partialDerivativeShape);
        }

        return new PartialDerivatives(getKey(), slicedPartialDerivative);
    }

    private long[] extractWrtShape(long[] partialDerivativeShape, int rankOfSource) {
        return extractShape(partialDerivativeShape, rankOfSource, rankOfSource, partialDerivativeShape.length);
    }

    private long[] extractOfShape(long[] partialDerivativeShape, int rankOfSource) {
        return extractShape(partialDerivativeShape, rankOfSource, 0, rankOfSource);
    }

    private long[] extractShape(long[] partialDerivativeShape, int rankOfSource, int from, int to) {
        if (partialDerivativeShape.length == 0) {
            if (rankOfSource > 1) {
                throw new IllegalArgumentException("Partial does not contain of shape requested");
            } else {
                return new long[0];
            }
        }
        return Arrays.copyOfRange(partialDerivativeShape, from, to);
    }

    private static DoubleTensor increaseRankByAppendingOnesToShape(DoubleTensor lowRankTensor, int desiredRank) {
        return lowRankTensor.reshape(
            TensorShape.shapeDesiredToRankByAppendingOnes(lowRankTensor.getShape(), desiredRank)
        );
    }

    private static DoubleTensor increaseRankByPrependingOnesToShape(DoubleTensor lowRankTensor, int desiredRank) {
        return lowRankTensor.reshape(
            TensorShape.shapeToDesiredRankByPrependingOnes(lowRankTensor.getShape(), desiredRank)
        );
    }
}
