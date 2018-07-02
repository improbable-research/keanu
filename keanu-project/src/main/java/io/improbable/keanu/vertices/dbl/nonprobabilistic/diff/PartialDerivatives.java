package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static java.util.Collections.singletonMap;

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

    public PartialDerivatives sum(int... overDimensions) {
        Map<Long, DoubleTensor> summed = cloneInfinitesimals(derivativeWithRespectTo);

        for (Map.Entry<Long, DoubleTensor> entry : derivativeWithRespectTo.entrySet()) {
            long k = entry.getKey();
            DoubleTensor v = entry.getValue();
            DoubleTensor reshapedV = v.sum(overDimensions);
            summed.put(k, increaseRankByPrependingOnesToShape(reshapedV, v.getRank()));
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

            int[] partialWrtShape = Arrays.copyOfRange(partial.getShape(), multiplier.getRank(), partial.getRank());

            return partial.tensorMultiply(multiplierReshaped,
                range(0, partialOfShape.length),
                range(multiplier.getRank(), partial.getRank())
            ).reshape(TensorShape.concat(multiplier.getShape(), partialWrtShape));
        } else {
            return partial.times(multiplierReshaped);
        }
    }

    private static int[] range(int from, int to) {
        int[] result = new int[to - from];
        for (int i = 0; i < result.length; i++) {
            result[i] = from + i;
        }
        return result;
    }

    private static DoubleTensor increaseRankByAppendingOnesToShape(DoubleTensor lowRankTensor, int desiredRank) {
        return increaseRankByPaddingOnes(lowRankTensor, desiredRank, true);
    }

    private static DoubleTensor increaseRankByPrependingOnesToShape(DoubleTensor lowRankTensor, int desiredRank) {
        return increaseRankByPaddingOnes(lowRankTensor, desiredRank, false);
    }

    private static DoubleTensor increaseRankByPaddingOnes(DoubleTensor lowRankTensor, int desiredRank, boolean append) {
        int[] shape = lowRankTensor.getShape();
        if (shape.length == desiredRank) {
            return lowRankTensor;
        }

        int[] paddedShape = new int[desiredRank];
        Arrays.fill(paddedShape, 1);
        if (append) {
            System.arraycopy(shape, 0, paddedShape, 0, shape.length);
        } else {
            System.arraycopy(shape, 0, paddedShape, shape.length, shape.length);
        }
        return lowRankTensor.reshape(paddedShape);
    }

    public static PartialDerivatives matrixMultiply(PartialDerivatives partials, DoubleTensor multiplier, boolean partialIsLeft) {
        Map<Long, DoubleTensor> multiplied = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> partial : partials.derivativeWithRespectTo.entrySet()) {

            DoubleTensor reshapedMultiplier = increaseRankByAppendingOnesToShape(multiplier, partial.getValue().getRank());
            int[] partialShape = partial.getValue().getShape();
            int[] resultShape = Arrays.copyOf(partialShape, partialShape.length);

            DoubleTensor v;
            if (partialIsLeft) {
                resultShape[0] = partialShape[0];
                resultShape[1] = multiplier.getShape()[1];
                v = partial.getValue()
                    .tensorMultiply(reshapedMultiplier, new int[]{1}, new int[]{0})
                    .reshape(-1, resultShape[1])
                    .transpose()
                    .reshape(resultShape);
            } else {
                resultShape[0] = multiplier.getShape()[0];
                resultShape[1] = partialShape[1];
                v = reshapedMultiplier
                    .tensorMultiply(partial.getValue(), new int[]{1}, new int[]{0})
                    .reshape(resultShape);
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

    private static Map<Long, DoubleTensor> cloneInfinitesimals(Map<Long, DoubleTensor> infinitesimals) {
        Map<Long, DoubleTensor> clone = new HashMap<>();
        for (Map.Entry<Long, DoubleTensor> entry : infinitesimals.entrySet()) {
            clone.put(entry.getKey(), entry.getValue());
        }
        return clone;
    }
}
