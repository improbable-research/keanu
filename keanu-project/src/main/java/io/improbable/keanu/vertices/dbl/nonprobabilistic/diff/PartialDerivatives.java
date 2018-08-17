package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import static java.util.Collections.singletonMap;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

public class PartialDerivatives {

    public static PartialDerivatives OF_CONSTANT = new PartialDerivatives(Collections.emptyMap());

    public static PartialDerivatives withRespectToSelf(long withRespectTo, int[] shape) {
        return new PartialDerivatives(
            singletonMap(
                withRespectTo,
                DoubleTensor.eye((int) TensorShape.getLength(shape)).reshape(TensorShape.concat(shape, shape))
            )
        );
    }

    public static PartialDerivatives ifThenElse(BooleanTensor predicate, PartialDerivatives thn, PartialDerivatives els) {
        DoubleTensor trueMask = predicate.toDoubleMask();
        DoubleTensor falseMask = predicate.not().toDoubleMask();

        Map<Long, DoubleTensor> thenPartials = thn.derivativeWithRespectTo;
        Map<Long, DoubleTensor> elsePartials = els.derivativeWithRespectTo;
        Set<Long> wrtUnion = new HashSet<>();
        wrtUnion.addAll(thenPartials.keySet());
        wrtUnion.addAll(elsePartials.keySet());

        Map<Long, DoubleTensor> mixedPartials = new HashMap<>();
        for (Long wrt : wrtUnion) {
            DoubleTensor thnPartial = thenPartials.get(wrt);
            DoubleTensor elsPartial = elsePartials.get(wrt);
            DoubleTensor broadcastedTrueMask;
            DoubleTensor broadcastedFalseMask;

            DoubleTensor newPartial;
            if (thnPartial == null) {
                broadcastedFalseMask = DoubleTensor.zeros(elsPartial.getShape()).plusInPlace(falseMask);
                newPartial = broadcastedFalseMask.timesInPlace(elsPartial);
            } else if (elsPartial == null) {
                broadcastedTrueMask = DoubleTensor.zeros(thnPartial.getShape()).plusInPlace(trueMask);
                newPartial = broadcastedTrueMask.timesInPlace(thnPartial);
            } else {
                broadcastedFalseMask = DoubleTensor.zeros(thnPartial.getShape()).plusInPlace(falseMask);
                broadcastedTrueMask = DoubleTensor.zeros(thnPartial.getShape()).plusInPlace(trueMask);

                newPartial = broadcastedTrueMask.timesInPlace(thnPartial)
                    .plusInPlace(broadcastedFalseMask.timesInPlace(elsPartial));
            }

            mixedPartials.put(wrt, newPartial);
        }

        return new PartialDerivatives(mixedPartials);
    }

    private Map<Long, DoubleTensor> derivativeWithRespectTo;

    public PartialDerivatives(long id, DoubleTensor derivativeWithRespectTo) {
        this.derivativeWithRespectTo = new HashMap<>();
        this.derivativeWithRespectTo.put(id, derivativeWithRespectTo);
    }

    public PartialDerivatives(Map<Long, DoubleTensor> derivativeWithRespectTo) {
        this.derivativeWithRespectTo = derivativeWithRespectTo;
    }

    public DoubleTensor withRespectTo(Vertex vertex) {
        return withRespectTo(vertex.getId());
    }

    public DoubleTensor withRespectTo(long id) {
        return derivativeWithRespectTo.getOrDefault(id, DoubleTensor.ZERO_SCALAR);
    }

    public boolean isEmpty() {
        return derivativeWithRespectTo.isEmpty();
    }

    public Map<Long, DoubleTensor> asMap() {
        return derivativeWithRespectTo;
    }

    public void putWithRespectTo(long id, DoubleTensor value) {
        derivativeWithRespectTo.put(id, value);
    }

    /**
     * This will sum partial derivatives that are represented as tensors over given dimensions.
     * There is the option to reshape to a lower rank tensor where the summation has caused a
     * dimension to go to length 1.
     *
     * @param reshape        Returns the sum and drops the summed over dimensions (now length one)
     *                       in the shape if true. Returns a same ranked tensor but with a shape
     *                       that has ones for the dimensions summed over.
     * @param overDimensions The dimensions to sum over. Dimensions are counted from zero
     * @return The summed partial derivatives over given dimensions
     */
    public PartialDerivatives sum(boolean reshape, int... overDimensions) {
        Map<Long, DoubleTensor> summed = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue();
            DoubleTensor reshapedV = v.sum(overDimensions);
            if (reshape) {
                summed.put(k, reshapedV);
            } else {
                summed.put(k, increaseRankByPrependingOnesToShape(reshapedV, v.getRank()));
            }
        }

        return new PartialDerivatives(summed);
    }

    public PartialDerivatives add(PartialDerivatives toAdd) {
        Map<Long, DoubleTensor> added = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<Long, DoubleTensor> entry : toAdd.derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (added.containsKey(k)) {
                added.put(k, added.get(k).plus(v));
            } else {
                added.put(k, v);
            }
        }

        return new PartialDerivatives(added);
    }

    public PartialDerivatives subtract(PartialDerivatives toSubtract) {
        Map<Long, DoubleTensor> subtracted = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<Long, DoubleTensor> entry : toSubtract.derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (subtracted.containsKey(k)) {
                subtracted.put(k, subtracted.get(k).minus(v));
            } else {
                subtracted.put(k, v.unaryMinus());
            }
        }

        return new PartialDerivatives(subtracted);
    }

    public PartialDerivatives multiplyBy(DoubleTensor multiplier) {
        Map<Long, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = elementWiseMultiplyDiff(entry.getValue(), multiplier);
            multiplied.put(k, v);
        }

        return new PartialDerivatives(multiplied);
    }

    private DoubleTensor elementWiseMultiplyDiff(DoubleTensor partial, DoubleTensor multiplier) {

        if (multiplier.isScalar()) {
            return partial.times(multiplier.scalar());
        }

        DoubleTensor multiplierReshaped = increaseRankByAppendingOnesToShape(multiplier, partial.getRank());

        if (partial.isScalar()) {
            return multiplierReshaped.times(partial.scalar());
        }

        int[] partialOfShape = Arrays.copyOfRange(partial.getShape(), 0, multiplier.getRank());

        if (TensorShape.isScalar(partialOfShape)) {

            int[] partialWrtShape = extractWrtShape(partial.getShape(), multiplier.getRank());

            return partial.tensorMultiply(multiplierReshaped,
                TensorShape.dimensionRange(0, partialOfShape.length),
                TensorShape.dimensionRange(multiplier.getRank(), partial.getRank())
            ).reshape(TensorShape.concat(multiplier.getShape(), partialWrtShape));
        } else {
            return partial.times(multiplierReshaped);
        }
    }

    public static PartialDerivatives matrixMultiply(PartialDerivatives partials, DoubleTensor multiplier, boolean partialIsLeft) {
        Map<Long, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> partial : partials.derivativeWithRespectTo.entrySet()) {

            int partialRank = partial.getValue().getRank();

            DoubleTensor v;
            if (partialIsLeft) {
                int[] rearrange = TensorShape.dimensionRange(-1, partialRank - 1);
                rearrange[0] = 0;
                rearrange[1] = partialRank - 1;
                v = partial.getValue()
                    .tensorMultiply(multiplier, new int[]{1}, new int[]{0})
                    .permute(rearrange);

            } else {
                v = multiplier
                    .tensorMultiply(partial.getValue(), new int[]{1}, new int[]{0});
            }
            multiplied.put(partial.getKey(), v);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives multiplyBy(double multiplier) {
        Map<Long, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().times(multiplier);
            multiplied.put(k, v);
        }

        return new PartialDerivatives(multiplied);
    }

    public PartialDerivatives divideBy(DoubleTensor divisor) {
        Map<Long, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor partial = entry.getValue();
            DoubleTensor v = partial.div(increaseRankByAppendingOnesToShape(divisor, partial.getRank()));
            divided.put(k, v);
        }

        return new PartialDerivatives(divided);
    }

    public PartialDerivatives divideBy(double divisor) {
        Map<Long, DoubleTensor> divided = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().div(divisor);
            divided.put(k, v);
        }

        return new PartialDerivatives(divided);
    }

    public PartialDerivatives powerTo(double power) {
        Map<Long, DoubleTensor> powered = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue().pow(power);
            powered.put(k, v);
        }

        return new PartialDerivatives(powered);
    }

    public PartialDerivatives clone() {
        return new PartialDerivatives(cloneInfinitesimals(derivativeWithRespectTo));
    }

    public PartialDerivatives reshape(int currentRank, int[] proposedShape) {
        Map<Long, DoubleTensor> reshapedDerivatives = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> partialDerivative : derivativeWithRespectTo.entrySet()) {
            int[] shape = partialDerivative.getValue().getShape();
            int[] wrtShape = extractWrtShape(shape, currentRank);
            int[] newPartialShape = TensorShape.concat(proposedShape, wrtShape);

            DoubleTensor reshapedPartialDerivative = partialDerivative.getValue().reshape(newPartialShape);
            reshapedDerivatives.put(partialDerivative.getKey(), reshapedPartialDerivative);
        }

        return new PartialDerivatives(reshapedDerivatives);
    }

    public PartialDerivatives slice(int dimension, int index) {
        Map<Long, DoubleTensor> slicedDerivatives = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> partialDerivative : derivativeWithRespectTo.entrySet()) {
            int[] partialDerivativeShape = partialDerivative.getValue().getShape();
            partialDerivativeShape[dimension] = 1;
            DoubleTensor slicedPartialDerivative = partialDerivative.getValue().slice(dimension, index);
            slicedPartialDerivative = slicedPartialDerivative.reshape(partialDerivativeShape);
            slicedDerivatives.put(partialDerivative.getKey(), slicedPartialDerivative);
        }

        return new PartialDerivatives(slicedDerivatives);
    }

    private static Map<Long, DoubleTensor> cloneInfinitesimals(Map<Long, DoubleTensor> infinitesimals) {
        Map<Long, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<Long, DoubleTensor> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }

    private int[] extractWrtShape(int[] partialDerivativeShape, int rankOfSource) {
        int[] wrtShape = Arrays.copyOfRange(partialDerivativeShape, rankOfSource, partialDerivativeShape.length);
        return wrtShape;
    }

    public static DoubleTensor increaseRankByPrependingOnesToShape(DoubleTensor lowRankTensor, int desiredRank) {
        return lowRankTensor.reshape(
            TensorShape.shapeToDesiredRankByPrependingOnes(lowRankTensor.getShape(), desiredRank)
        );
    }

    public static DoubleTensor increaseRankByAppendingOnesToShape(DoubleTensor lowRankTensor, int desiredRank) {
        return lowRankTensor.reshape(
            TensorShape.shapeDesiredToRankByAppendingOnes(lowRankTensor.getShape(), desiredRank)
        );
    }

}
