package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import static java.util.Collections.singletonMap;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.Nd4jDoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;

public class PartialDerivatives {

    public static PartialDerivatives OF_CONSTANT = new PartialDerivatives(Collections.emptyMap());

    public static PartialDerivatives withRespectToSelf(VertexId withRespectTo, int[] shape) {
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

        Map<VertexId, DoubleTensor> thenPartials = thn.derivativeWithRespectTo;
        Map<VertexId, DoubleTensor> elsePartials = els.derivativeWithRespectTo;
        Set<VertexId> wrtUnion = new HashSet<>();
        wrtUnion.addAll(thenPartials.keySet());
        wrtUnion.addAll(elsePartials.keySet());

        Map<VertexId, DoubleTensor> mixedPartials = new HashMap<>();
        for (VertexId wrt : wrtUnion) {
            DoubleTensor thnPartial = thenPartials.get(wrt);
            DoubleTensor elsPartial = elsePartials.get(wrt);
            DoubleTensor broadcastedTrueMask;
            DoubleTensor broadcastedFalseMask;
            int[] range = TensorShape.dimensionRange(0, thnPartial == null ? elsPartial.getRank() : thnPartial.getRank());
            int[] permute = TensorShape.concat(
                Arrays.copyOfRange(range, range.length - 2, range.length),
                Arrays.copyOfRange(range, 0, range.length - 2)
            );

            DoubleTensor newPartial;
            if (thnPartial == null) {
                broadcastedFalseMask = DoubleTensor.zeros(elsPartial.getShape()).plusInPlace(falseMask).permute(permute);
                newPartial = broadcastedFalseMask.timesInPlace(elsPartial);
            } else if (elsPartial == null) {
                broadcastedTrueMask = DoubleTensor.zeros(thnPartial.getShape()).plusInPlace(trueMask).permute(permute);
                newPartial = broadcastedTrueMask.timesInPlace(thnPartial);
            } else {
                broadcastedFalseMask = DoubleTensor.zeros(thnPartial.getShape()).plusInPlace(falseMask).permute(permute);
                broadcastedTrueMask = DoubleTensor.zeros(thnPartial.getShape()).plusInPlace(trueMask).permute(permute);

                newPartial = broadcastedTrueMask.timesInPlace(thnPartial)
                    .plusInPlace(broadcastedFalseMask.timesInPlace(elsPartial));
            }

            mixedPartials.put(wrt, newPartial);
        }

        return new PartialDerivatives(mixedPartials);
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
        Map<VertexId, DoubleTensor> summed = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
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
        return add(toAdd, new HashMap<>());
    }

    public PartialDerivatives add(PartialDerivatives toAdd, Map<VertexId, List<Integer>> reshapes) {
        Map<VertexId, DoubleTensor> added = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();
            if (reshapes.containsKey(k)) {
                int[] desiredShape = reshapes.get(k).stream().mapToInt(i -> i).toArray();
                added.put(k, Nd4jDoubleTensor.ones(desiredShape).times(v));
            } else {
                added.put(k, v);
            }
        }

        for (Map.Entry<VertexId, DoubleTensor> entry : toAdd.derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue();

            if (added.containsKey(k)) {
                added.put(k, added.get(k).plus(v));
            } else {
                if (reshapes.containsKey(k)) {
                    int[] desiredShape = reshapes.get(k).stream().mapToInt(i -> i).toArray();
                    added.put(k, Nd4jDoubleTensor.ones(desiredShape).times(v));
                } else {
                    added.put(k, v);
                }
            }
        }

        return new PartialDerivatives(added);
    }

    public PartialDerivatives subtract(PartialDerivatives toSubtract) {
        Map<VertexId, DoubleTensor> subtracted = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<VertexId, DoubleTensor> entry : toSubtract.derivativeWithRespectTo.entrySet()) {
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

    public PartialDerivatives multiplyBy(DoubleTensor multiplier) {
        Map<VertexId, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
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
        Map<VertexId, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partial : partials.derivativeWithRespectTo.entrySet()) {

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

    public static PartialDerivatives matrixMultiplyReverse(PartialDerivatives partials, DoubleTensor multiplier, boolean partialIsLeft) {
        Map<VertexId, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partial : partials.derivativeWithRespectTo.entrySet()) {

            int partialRank = partial.getValue().getRank();

            DoubleTensor v;
            if (partialIsLeft) {
                v = partial.getValue()
                    .tensorMultiply(multiplier, new int[]{partialRank - 1}, new int[]{1});
            } else {
                int[] rearrange = TensorShape.dimensionRange(0, partialRank);
                rearrange[partialRank - 1] = partialRank - 2;
                rearrange[partialRank - 2] = partialRank - 1;

                v = partial.getValue()
                    .tensorMultiply(multiplier, new int[]{partialRank - 2}, new int[]{0})
                    .permute(rearrange);
            }
            multiplied.put(partial.getKey(), v);
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

    public PartialDerivatives powerTo(double power) {
        Map<VertexId, DoubleTensor> powered = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            VertexId k = entry.getKey();
            DoubleTensor v = entry.getValue().pow(power);
            powered.put(k, v);
        }

        return new PartialDerivatives(powered);
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

    public PartialDerivatives slice(int dimension, int index) {
        Map<VertexId, DoubleTensor> slicedDerivatives = new HashMap<>();

        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeWithRespectTo.entrySet()) {
            int[] partialDerivativeShape = partialDerivative.getValue().getShape();
            partialDerivativeShape[dimension] = 1;
            DoubleTensor slicedPartialDerivative = partialDerivative.getValue().slice(dimension, index);
            slicedPartialDerivative = slicedPartialDerivative.reshape(partialDerivativeShape);
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
