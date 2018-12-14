package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import org.nd4j.linalg.api.shape.Shape;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static java.util.Collections.singletonMap;

public class PartialDerivatives {

    public static final PartialDerivatives OF_CONSTANT = new PartialDerivatives(Collections.emptyMap());

    public static PartialDerivatives withRespectToSelf(VertexId withRespectTo, long[] shape) {
        return new PartialDerivatives(
            singletonMap(
                withRespectTo,
                DoubleTensor.eye((int) TensorShape.getLength(shape)).reshape(TensorShape.concat(shape, shape))
            )
        );
    }

    private Map<VertexId, DoubleTensor> derivativeWithRespectTo;

    public PartialDerivatives(VertexId id, DoubleTensor derivativeWithRespectTo) {
        this.derivativeWithRespectTo = new HashMap<>();
        this.derivativeWithRespectTo.put(id, derivativeWithRespectTo);
    }

    public PartialDerivatives(Map<VertexId, DoubleTensor> derivativeWithRespectTo) {
        this.derivativeWithRespectTo = derivativeWithRespectTo;
        if (derivativeWithRespectTo.size() > 1) {
            throw new IllegalStateException();
        }
    }

    public DoubleTensor withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public DoubleTensor withRespectTo(VertexId id) {
        return derivativeWithRespectTo.getOrDefault(id, DoubleTensor.ZERO_SCALAR);
    }

    public boolean isEmpty() {
        return derivativeWithRespectTo.isEmpty();
    }

    public void putWithRespectTo(VertexId id, DoubleTensor value) {
        derivativeWithRespectTo.put(id, value);
        if (derivativeWithRespectTo.size() > 1) {
            throw new IllegalStateException();
        }
    }

    public DoubleTensor getValue() {
        if (this.derivativeWithRespectTo.size() > 1) {
            throw new IllegalStateException();
        }
        VertexId valueId = getKey();

        return derivativeWithRespectTo.get(valueId);
    }

    public VertexId getKey() {
        if (derivativeWithRespectTo.isEmpty()) {
            return null;
        }

        return derivativeWithRespectTo.keySet().iterator().next();
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
        Map<VertexId, DoubleTensor> summed = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();
            long[] vShape = v.getShape();
            long[] wrtShape = TensorShape.selectDimensions(ofRank, vShape.length, vShape);

            DoubleTensor summedV = v.sum(dimensions);
            long[] newShape = TensorShape.concat(resultShape, wrtShape);
            summedV = summedV.reshape(newShape);

            summed.put(k, summedV);
        }

        return new PartialDerivatives(summed);
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
        Map<VertexId, DoubleTensor> summed = cloneInfinitesimals(derivativeWithRespectTo);

        if (dimensions.length == 0) {
            return new PartialDerivatives(summed);
        }

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();
            long[] vShape = v.getShape();
            long[] ofShape = TensorShape.selectDimensions(0, v.getShape().length - wrtRank, vShape);

            DoubleTensor summedV = v.sum(dimensions);
            long[] newShape = TensorShape.concat(ofShape, resultShape);
            summedV = summedV.reshape(newShape);

            summed.put(k, summedV);
        }

        return new PartialDerivatives(summed);
    }

    public PartialDerivatives add(PartialDerivatives toAdd) {
        return add(toAdd, false, false, null);
    }

    public PartialDerivatives add(PartialDerivatives addition, boolean leftIsLengthOne, boolean rightIsLengthOne, long[] resultShape) {

        Map<VertexId, DoubleTensor> added = cloneWithCorrectShape(derivativeWithRespectTo, leftIsLengthOne, resultShape);
        Map<VertexId, DoubleTensor> toAdd = cloneWithCorrectShape(addition.derivativeWithRespectTo, rightIsLengthOne, resultShape);

        for (Map.Entry<VertexId, DoubleTensor> entry : toAdd.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (added.containsKey(k)) {
                added.put(k, added.get(k).plus(v));
            } else {
                added.put(k, v);
            }
        }

        return new PartialDerivatives(added);
    }

    public PartialDerivatives subtract(PartialDerivatives subtraction) {
        return subtract(subtraction, false, false, null);
    }

    public PartialDerivatives subtract(PartialDerivatives subtraction, boolean leftIsLengthOne, boolean rightIsLengthOne, long[] resultShape) {

        Map<VertexId, DoubleTensor> subtracted = cloneWithCorrectShape(derivativeWithRespectTo, leftIsLengthOne, resultShape);
        Map<VertexId, DoubleTensor> toSubtract = cloneWithCorrectShape(subtraction.derivativeWithRespectTo, rightIsLengthOne, resultShape);

        for (Map.Entry<VertexId, DoubleTensor> entry : toSubtract.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (subtracted.containsKey(k)) {
                subtracted.put(k, subtracted.get(k).minus(v));
            } else {
                subtracted.put(k, v.unaryMinus());
            }
        }

        return new PartialDerivatives(subtracted);
    }

    private static Map<VertexId, DoubleTensor> cloneWithCorrectShape(Map<VertexId, DoubleTensor> infinitesimals,
                                                                     boolean ofIsLengthOne,
                                                                     long[] resultShape) {

        Map<VertexId, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<VertexId, DoubleTensor> entry : infinitesimals.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (ofIsLengthOne) {
                v = DoubleTensor.zeros(TensorShape.concat(resultShape, v.getShape())).plus(v);
            }

            clone.put(k, v);
        }
        return clone;
    }

    public PartialDerivatives multiplyAlongOfDimensions(DoubleTensor multiplier, long[] ofShape) {
        Map<VertexId, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor partial = entry.getValue();
            DoubleTensor result;

            if (multiplier.isScalar()) {
                result = partial.times(multiplier.scalar());
            } else {
                result = elementWiseMultiplyAlongOf(partial, multiplier, ofShape);
            }

            multiplied.put(k, result);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives multiplyAlongWrtDimensions(DoubleTensor multiplier, long[] wrtShape) {
        Map<VertexId, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor partial = entry.getValue();
            DoubleTensor result;

            if (multiplier.isScalar()) {
                result = partial.times(multiplier.scalar());
            } else {
                result = elementWiseMultiplyAlongWrt(partial, multiplier, wrtShape);
            }

            multiplied.put(k, result);
        }

        return new PartialDerivatives(multiplied);
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

    public static PartialDerivatives matrixMultiplyAlongOfDimensions(PartialDerivatives partials, DoubleTensor multiplier, boolean partialIsLeft) {
        Map<VertexId, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partial : partials.derivativeWithRespectTo.entrySet()) {

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
            multiplied.put(partial.getKey(), result);
        }

        return new PartialDerivatives(multiplied);
    }

    public static PartialDerivatives matrixMultiplyAlongWrtDimensions(PartialDerivatives partials, DoubleTensor multiplier, boolean partialIsLeft) {
        Map<VertexId, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partial : partials.derivativeWithRespectTo.entrySet()) {

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
            multiplied.put(partial.getKey(), result);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives multiplyBy(double multiplier) {
        Map<VertexId, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue().times(multiplier);
            multiplied.put(k, v);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives divideBy(DoubleTensor divisor) {
        Map<VertexId, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor partial = entry.getValue();

            DoubleTensor v = partial.div(increaseRankByAppendingOnesToShape(divisor, partial.getRank()));
            divided.put(k, v);
        }

        return new PartialDerivatives(divided);
    }

    public PartialDerivatives divideBy(double divisor) {
        Map<VertexId, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new PartialDerivatives(divided);
    }

    public PartialDerivatives clone() {
        return new PartialDerivatives(cloneInfinitesimals(derivativeWithRespectTo));
    }

    public PartialDerivatives reshape(int currentRank, long[] proposedShape) {
        Map<VertexId, DoubleTensor> reshapedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeWithRespectTo.entrySet()) {
            long[] shape = partialDerivative.getValue().getShape();
            long[] wrtShape = extractWrtShape(shape, currentRank);
            long[] newPartialShape = TensorShape.concat(proposedShape, wrtShape);

            DoubleTensor reshapedPartialDerivative = partialDerivative.getValue().reshape(newPartialShape);
            reshapedDerivatives.put(partialDerivative.getKey(), reshapedPartialDerivative);
        }

        return new PartialDerivatives(reshapedDerivatives);
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
        Map<VertexId, DoubleTensor> slicedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeWithRespectTo.entrySet()) {
            long[] partialDerivativeShape = Arrays.copyOf(partialDerivative.getValue().getShape(), partialDerivative.getValue().getShape().length);
            partialDerivativeShape[dimension] = 1;
            DoubleTensor slicedPartialDerivative = partialDerivative.getValue().slice(dimension, index);

            if (reshape) {
                slicedPartialDerivative = slicedPartialDerivative.reshape(partialDerivativeShape);
            }
            slicedDerivatives.put(partialDerivative.getKey(), slicedPartialDerivative);
        }

        return new PartialDerivatives(slicedDerivatives);
    }

    private static Map<VertexId, DoubleTensor> cloneInfinitesimals(Map<VertexId, DoubleTensor> infinitesimals) {
        Map<VertexId, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<VertexId, DoubleTensor> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
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
