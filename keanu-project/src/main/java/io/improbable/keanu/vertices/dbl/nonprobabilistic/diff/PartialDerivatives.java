package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import static java.util.Collections.singletonMap;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;

public class PartialDerivatives {

    public static final PartialDerivatives OF_CONSTANT = new PartialDerivatives(Collections.emptyMap());

    public static PartialDerivatives withRespectToSelf(VertexId withRespectTo, int[] shape) {
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

    public Map<VertexId, DoubleTensor> asMap() {
        return derivativeWithRespectTo;
    }

    public void putWithRespectTo(VertexId id, DoubleTensor value) {
        derivativeWithRespectTo.put(id, value);
    }

    /**
     * This will sum partial derivatives that are represented as tensors over given dimensions.
     * The dimensions that are summed over will be reshaped to a scalar shape of 1x1.
     *
     * @param resultShape
     * @param sumOfRank   the rank of the of part of the partials
     * @return summed and reshaped partials
     */
    public PartialDerivatives sumOverOfDimensions(int[] sumOverDimensions, int[] resultShape, int sumOfRank) {
        Map<VertexId, DoubleTensor> summed = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();
            DoubleTensor summedV = v.sum(sumOverDimensions);
            int[] vShape = v.getShape();
            int[] wrtShape = TensorShape.selectDimensions(sumOfRank, vShape.length - 1, vShape);

            int[] newShape = TensorShape.concat(resultShape, wrtShape);
            summedV = summedV.reshape(newShape);

            summed.put(k, summedV);
        }

        return new PartialDerivatives(summed);
    }

    /**
     * This will sum partial derivatives that are represented as tensors over given dimensions.
     * The dimensions that are summed over will be reshaped to a scalar shape of 1x1.
     *
     * @param wrtDimensions dimensions to sum over (should be last n dimensions)
     * @return summed and reshaped partials
     */
    public PartialDerivatives sumOverWrtDimensions(int... wrtDimensions) {
        Map<VertexId, DoubleTensor> summed = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();
            DoubleTensor summedV = v.sum(wrtDimensions);
            int[] newShape = TensorShape.concat(summedV.getShape(), Tensor.SCALAR_SHAPE);
            summedV = summedV.reshape(newShape);

            summed.put(k, summedV);
        }

        return new PartialDerivatives(summed);
    }

    public PartialDerivatives add(PartialDerivatives toAdd) {
        return add(toAdd, null);
    }

    public PartialDerivatives add(PartialDerivatives addition, int[] ofShape) {

        Map<VertexId, DoubleTensor> added = cloneWithCorrectShape(derivativeWithRespectTo, ofShape);
        Map<VertexId, DoubleTensor> toAdd = cloneWithCorrectShape(addition.derivativeWithRespectTo, ofShape);

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
        return subtract(subtraction, null);
    }

    public PartialDerivatives subtract(PartialDerivatives subtraction, int[] ofShape) {

        Map<VertexId, DoubleTensor> subtracted = cloneWithCorrectShape(derivativeWithRespectTo, ofShape);
        Map<VertexId, DoubleTensor> toSubtract = cloneWithCorrectShape(subtraction.derivativeWithRespectTo, ofShape);

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

    private static Map<VertexId, DoubleTensor> cloneWithCorrectShape(Map<VertexId, DoubleTensor> infinitesimals, int[] ofShape) {

        Map<VertexId, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<VertexId, DoubleTensor> entry : infinitesimals.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (ofShape == null || ofShapeMatches(ofShape, v.getShape())) {
                clone.put(k, v);
            } else {
                clone.put(k, DoubleTensor.zeros(shapeWrtScalar(ofShape, v.getShape())).plus(v));
            }
        }
        return clone;
    }

    private static boolean ofShapeMatches(int[] ofShape, int[] partialShape) {
        for (int i = 0; i < ofShape.length; i++) {
            if (ofShape[i] != partialShape[i]) {
                return false;
            }
        }
        return true;
    }

    private static int[] shapeWrtScalar(int[] ofShape, int[] partialShape) {
        int[] fixedShape = Arrays.copyOf(partialShape, partialShape.length);
        System.arraycopy(ofShape, 0, fixedShape, 0, ofShape.length);
        return fixedShape;
    }

    public PartialDerivatives multiplyAlongOfDimensions(DoubleTensor multiplier, int[] ofShape) {
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

    public PartialDerivatives multiplyAlongWrtDimensions(DoubleTensor multiplier, int[] wrtShape) {
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

    private DoubleTensor elementWiseMultiplyAlongOf(DoubleTensor partial, DoubleTensor multiplier, int[] ofShape) {

        int[] partialOfShape = extractOfShape(partial.getShape(), ofShape.length);
//        !Arrays.equals(partialOfShape, multiplier.getShape())
        if (TensorShape.isScalar(partialOfShape)) {

            int[] partialWrtShape = extractWrtShape(partial.getShape(), ofShape.length);
            int[] resultShape = TensorShape.concat(multiplier.getShape(), partialWrtShape);

            DoubleTensor multiplierFromLeft = increaseRankByAppendingOnesToShape(multiplier, resultShape.length);
            DoubleTensor appropriateShapePartial = increaseRankByPrependingOnesToShape(partial, resultShape.length);

            return DoubleTensor.ones(resultShape).times(appropriateShapePartial).times(multiplierFromLeft);
        }

        DoubleTensor multiplierFromLeft = increaseRankByAppendingOnesToShape(multiplier, partial.getRank());
        return partial.times(multiplierFromLeft);
    }

    private DoubleTensor elementWiseMultiplyAlongWrt(DoubleTensor partial, DoubleTensor multiplier, int[] wrtShape) {

        int[] partialWrtShape = extractWrtShape(partial.getShape(), partial.getRank() - wrtShape.length);
        if (!Arrays.equals(partialWrtShape, multiplier.getShape())) {
            int[] partialOfShape = extractOfShape(partial.getShape(), partial.getRank() - wrtShape.length);
            int[] resultShape = TensorShape.concat(partialOfShape, multiplier.getShape());

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

    public PartialDerivatives reshape(int currentRank, int[] proposedShape) {
        Map<VertexId, DoubleTensor> reshapedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeWithRespectTo.entrySet()) {
            int[] shape = partialDerivative.getValue().getShape();
            int[] wrtShape = extractWrtShape(shape, currentRank);
            int[] newPartialShape = TensorShape.concat(proposedShape, wrtShape);

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
    public PartialDerivatives slice(int dimension, int index, boolean reshape) {
        Map<VertexId, DoubleTensor> slicedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeWithRespectTo.entrySet()) {
            int[] partialDerivativeShape = partialDerivative.getValue().getShape();
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

    private int[] extractWrtShape(int[] partialDerivativeShape, int rankOfSource) {
        return Arrays.copyOfRange(partialDerivativeShape, rankOfSource, partialDerivativeShape.length);
    }

    private int[] extractOfShape(int[] partialDerivativeShape, int rankOfSource) {
        return Arrays.copyOfRange(partialDerivativeShape, 0, rankOfSource);
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
