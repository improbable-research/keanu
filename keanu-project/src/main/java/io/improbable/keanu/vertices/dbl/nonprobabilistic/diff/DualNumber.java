package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.*;
import java.util.stream.Collectors;

public class DualNumber {

    public static DualNumber createConstant(DoubleTensor value) {
        return new DualNumber(value, PartialDerivatives.OF_CONSTANT);
    }

    public static DualNumber createWithRespectToSelf(long withRespectTo, DoubleTensor value) {
        return new DualNumber(value, PartialDerivatives.withRespectToSelf(withRespectTo, value.getShape()));
    }

    public static DualNumber ifThenElse(BooleanTensor predicate, DualNumber thn, DualNumber els){
        if (predicate.allTrue()) {
            return new DualNumber(thn.value, thn.getPartialDerivatives());
        } else if (predicate.allFalse()) {
            return new DualNumber(els.value, els.getPartialDerivatives());
        } else {
            PartialDerivatives mixedPartials = PartialDerivatives.ifThenElse(predicate, thn.getPartialDerivatives(), els.getPartialDerivatives());
            return new DualNumber(predicate.setDoubleIf(thn.value, els.value), mixedPartials);
        }
    }

    private DoubleTensor value;
    private PartialDerivatives partialDerivatives;

    public DualNumber(DoubleTensor value, PartialDerivatives partialDerivatives) {
        this.value = value;
        this.partialDerivatives = partialDerivatives;
    }

    public DualNumber(DoubleTensor value, Map<Long, DoubleTensor> partialDerivatives) {
        this(value, new PartialDerivatives(partialDerivatives));
    }

    public DualNumber(DoubleTensor value, long infinitesimalLabel) {
        this(value, new PartialDerivatives(Collections.singletonMap(infinitesimalLabel, DoubleTensor.ones(value.getShape()))));
    }

    public DoubleTensor getValue() {
        return value;
    }

    public PartialDerivatives getPartialDerivatives() {
        return partialDerivatives;
    }

    public boolean isOfConstant() {
        return partialDerivatives.isEmpty();
    }

    public DualNumber add(DualNumber that) {
        // dc = da + db;
        DoubleTensor newValue = this.value.plus(that.value);
        Map<Long, List<Integer>> reshapes = new HashMap<>();
        reshapes = reshapeScalarOperations(this.value, this.getPartialDerivatives(), newValue, reshapes);
        reshapes = reshapeScalarOperations(that.value, that.getPartialDerivatives(), newValue, reshapes);
        PartialDerivatives newInf = this.partialDerivatives.add(that.partialDerivatives, reshapes);
        return new DualNumber(newValue, newInf);
    }

    private Map<Long, List<Integer>> reshapeScalarOperations(DoubleTensor primary, PartialDerivatives partials, DoubleTensor newValue, Map<Long, List<Integer>> reshapedScalars) {
        if (primary.isScalar()) {
            for (Map.Entry<Long, DoubleTensor> partial : partials.asMap().entrySet()) {
                if (partial.getValue().isScalar()) {
                    int[] desiredShape = TensorShape.concat(newValue.getShape(), primary.getShape());
                    reshapedScalars.put(partial.getKey(), Arrays.stream(desiredShape).boxed().collect(Collectors.toList()));
                }
            }
        }
        return reshapedScalars;
    }

    public DualNumber subtract(DualNumber that) {
        // dc = da - db;
        return plus(that.unaryMinus());
    }

    public DualNumber matrixMultiplyBy(DualNumber that) {
        // dc = A * db + da * B;
        DoubleTensor newValue = this.value.matrixMultiply(that.value);
        PartialDerivatives thisInfMultiplied;
        PartialDerivatives thatInfMultiplied;

        if (this.partialDerivatives.isEmpty()) {
            thisInfMultiplied = PartialDerivatives.OF_CONSTANT;
        } else {
            thisInfMultiplied = PartialDerivatives.matrixMultiply(this.partialDerivatives, that.value, true);
        }

        if (that.partialDerivatives.isEmpty()) {
            thatInfMultiplied = PartialDerivatives.OF_CONSTANT;
        } else {
            thatInfMultiplied = PartialDerivatives.matrixMultiply(that.partialDerivatives, this.value, false);
        }

        PartialDerivatives newInf = thisInfMultiplied.add(thatInfMultiplied);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber multiplyBy(DualNumber that) {
        // dc = A * db + da * B;
        DoubleTensor newValue = this.value.times(that.value);
        PartialDerivatives thisInfMultiplied;
        PartialDerivatives thatInfMultiplied;

        if (this.partialDerivatives.isEmpty()) {
            thisInfMultiplied = PartialDerivatives.OF_CONSTANT;
        } else {
            thisInfMultiplied = this.partialDerivatives.multiplyBy(that.value);
        }

        if (that.partialDerivatives.isEmpty()) {
            thatInfMultiplied = PartialDerivatives.OF_CONSTANT;
        } else {
            thatInfMultiplied = that.partialDerivatives.multiplyBy(this.value);
        }

        PartialDerivatives newInf = thisInfMultiplied.add(thatInfMultiplied);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber divideBy(DualNumber that) {
        // dc = (B * da - A * db) / B^2;
        DoubleTensor newValue = this.value.div(that.value);
        PartialDerivatives thisInfMultiplied;
        PartialDerivatives thatInfMultiplied;
        PartialDerivatives newInf;

        if (this.partialDerivatives.isEmpty()) {
            thisInfMultiplied = PartialDerivatives.OF_CONSTANT;
        } else {
            thisInfMultiplied = this.partialDerivatives.multiplyBy(that.value);
        }

        if (that.partialDerivatives.isEmpty()) {
            thatInfMultiplied = PartialDerivatives.OF_CONSTANT;
        } else {
            thatInfMultiplied = that.partialDerivatives.multiplyBy(this.value);
        }

        if (thisInfMultiplied.isEmpty() && thatInfMultiplied.isEmpty()) {
            newInf = PartialDerivatives.OF_CONSTANT;
        } else {
            newInf = thisInfMultiplied.subtract(thatInfMultiplied).divideBy(that.value.pow(2));
        }

        return new DualNumber(newValue, newInf);
    }

    public DualNumber pow(DualNumber that) {
        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        DoubleTensor newValue = this.value.pow(that.value);
        PartialDerivatives thisInfBase;
        PartialDerivatives thisInfExponent;

        if (this.partialDerivatives.isEmpty()) {
            thisInfBase = PartialDerivatives.OF_CONSTANT;
        } else {
            thisInfBase = this.partialDerivatives.multiplyBy(that.value.times(this.value.pow(that.value.minus(1))));
        }

        if (that.partialDerivatives.isEmpty()) {
            thisInfExponent = PartialDerivatives.OF_CONSTANT;
        } else {
            thisInfExponent = that.partialDerivatives.multiplyBy(this.value.log().timesInPlace(newValue));
        }

        PartialDerivatives newInf = thisInfBase.add(thisInfExponent);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber plus(DualNumber that) {
        return add(that);
    }

    public DualNumber minus(DualNumber that) {
        return subtract(that);
    }

    public DualNumber times(DualNumber that) {
        return multiplyBy(that);
    }

    public DualNumber div(DualNumber that) {
        return divideBy(that);
    }

    public DualNumber plus(double value) {
        DoubleTensor newValue = this.value.plus(value);
        PartialDerivatives clonedInf = this.partialDerivatives.clone();
        return new DualNumber(newValue, clonedInf);
    }

    public DualNumber minus(double value) {
        DoubleTensor newValue = this.value.minus(value);
        PartialDerivatives clonedInf = this.partialDerivatives.clone();
        return new DualNumber(newValue, clonedInf);
    }

    public DualNumber times(double value) {
        DoubleTensor newValue = this.value.times(value);
        PartialDerivatives newInf = this.partialDerivatives.multiplyBy(value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber div(double value) {
        DoubleTensor newValue = this.value.div(value);
        PartialDerivatives newPartial = this.partialDerivatives.divideBy(value);
        return new DualNumber(newValue, newPartial);
    }

    public DualNumber unaryMinus() {
        return times(-1.0);
    }

    public DualNumber exp() {
        DoubleTensor newValue = value.exp();
        if (this.partialDerivatives.isEmpty()) {
            return new DualNumber(newValue, PartialDerivatives.OF_CONSTANT);
        } else {
            return new DualNumber(newValue, this.partialDerivatives.multiplyBy(newValue));
        }
    }

    public DualNumber sin() {
        DoubleTensor newValue = value.sin();
        if (this.partialDerivatives.isEmpty()) {
            return new DualNumber(newValue, PartialDerivatives.OF_CONSTANT);
        } else {
            DoubleTensor dSin = value.cos();
            return new DualNumber(newValue, this.partialDerivatives.multiplyBy(dSin));
        }
    }

    public DualNumber cos() {
        DoubleTensor newValue = value.cos();
        if (this.partialDerivatives.isEmpty()) {
            return new DualNumber(newValue, PartialDerivatives.OF_CONSTANT);
        } else {
            DoubleTensor dCos = value.sin().unaryMinusInPlace();
            return new DualNumber(newValue, this.partialDerivatives.multiplyBy(dCos));
        }
    }

    public DualNumber tan() {
        DoubleTensor newValue = value.tan();
        if (this.partialDerivatives.isEmpty()) {
            return new DualNumber(newValue, PartialDerivatives.OF_CONSTANT);
        } else {
            DoubleTensor dTan = value.cos().powInPlace(2).reciprocalInPlace();
            return new DualNumber(newValue, this.partialDerivatives.multiplyBy(dTan));
        }
    }

    public DualNumber asin() {
        DoubleTensor newValue = value.asin();
        if (this.partialDerivatives.isEmpty()) {
            return new DualNumber(newValue, PartialDerivatives.OF_CONSTANT);
        } else {
            DoubleTensor dArcSin = (value.unaryMinus().timesInPlace(value).plusInPlace(1))
                .sqrtInPlace().reciprocalInPlace();
            return new DualNumber(newValue, this.partialDerivatives.multiplyBy(dArcSin));
        }
    }

    public DualNumber acos() {
        DoubleTensor newValue = value.acos();
        if (this.partialDerivatives.isEmpty()) {
            return new DualNumber(newValue, PartialDerivatives.OF_CONSTANT);
        } else {
            DoubleTensor dArcCos = value.unaryMinus().timesInPlace(value).plusInPlace(1)
                .sqrtInPlace().reciprocalInPlace().unaryMinusInPlace();
            return new DualNumber(newValue, this.partialDerivatives.multiplyBy(dArcCos));
        }
    }

    public DualNumber atan() {
        DoubleTensor newValue = value.atan();
        if (this.partialDerivatives.isEmpty()) {
            return new DualNumber(newValue, PartialDerivatives.OF_CONSTANT);
        } else {
            DoubleTensor dArcTan = value.pow(2).plusInPlace(1).reciprocalInPlace();
            return new DualNumber(newValue, this.partialDerivatives.multiplyBy(dArcTan));
        }
    }

    public DualNumber log() {
        DoubleTensor newValue = value.log();
        if (this.partialDerivatives.isEmpty()) {
            return new DualNumber(newValue, PartialDerivatives.OF_CONSTANT);
        } else {
            return new DualNumber(newValue, this.partialDerivatives.divideBy(value));
        }
    }

    public DualNumber sum() {
        DoubleTensor sumOfAll = DoubleTensor.scalar(value.sum());
        int[] resultDims = TensorShape.dimensionRange(0, value.getRank());
        return new DualNumber(sumOfAll, this.partialDerivatives.sum(false, resultDims));
    }

    public DualNumber reshape(int[] proposedShape) {
        PartialDerivatives reshapedPartialDerivatives = this.partialDerivatives.reshape(getValue().getRank(), proposedShape);
        return new DualNumber(value.reshape(proposedShape), reshapedPartialDerivatives);
    }

    public DualNumber slice(int dimension, int index) {
        PartialDerivatives slicedPartialDerivatives = this.partialDerivatives.slice(dimension, index);
        return new DualNumber(value.slice(dimension, index), slicedPartialDerivatives);
    }

    public DualNumber concat(int dimension, List<DualNumber> dualToConcat, DoubleTensor... toConcat) {
        Map<Long, DoubleTensor> concatenatedPartialDerivatives = new HashMap<>();
        Map<Long, List<DoubleTensor>> combinedPartialDerivativesOfInputs = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> partial : this.partialDerivatives.asMap().entrySet()) {
            combinedPartialDerivativesOfInputs.computeIfAbsent(partial.getKey(), k -> new ArrayList<>()).add(partial.getValue());
        }

        for (int i = 0; i < dualToConcat.size(); i++) {
            for (Map.Entry<Long, DoubleTensor> partial : dualToConcat.get(i).getPartialDerivatives().asMap().entrySet()) {
                combinedPartialDerivativesOfInputs.computeIfAbsent(partial.getKey(), k -> new ArrayList<>()).add(partial.getValue());
            }
        }

        for (Map.Entry<Long, List<DoubleTensor>> partials : combinedPartialDerivativesOfInputs.entrySet()) {
            concatenatedPartialDerivatives.put(partials.getKey(), concatPartialDerivates(dimension, partials.getValue()));
        }

        DoubleTensor concatValue = this.getValue().concat(dimension, toConcat);
        return new DualNumber(concatValue, concatenatedPartialDerivatives);

    }

    private DoubleTensor concatPartialDerivates(int dimension, List<DoubleTensor> partialDerivates) {
        if (partialDerivates.size() == 1) {
            return partialDerivates.get(0);
        } else {
            DoubleTensor primaryTensor = partialDerivates.remove(0);
            DoubleTensor[] derivativesToConcat = new DoubleTensor[partialDerivates.size()];
            return primaryTensor.concat(dimension, partialDerivates.toArray(derivativesToConcat));
        }
    }

    public DualNumber take(int... index) {
        Map<Long, DoubleTensor> dualsAtIndex = new HashMap<>();

        for (Map.Entry<Long, DoubleTensor> entry : this.partialDerivatives.asMap().entrySet()) {
            DoubleTensor atIndexTensor = takeFromPartial(entry.getValue(), index);
            int desiredRank = entry.getValue().getShape().length;
            int[] paddedShape = TensorShape.shapeToDesiredRankByPrependingOnes(atIndexTensor.getShape(), desiredRank);
            atIndexTensor = atIndexTensor.reshape(paddedShape);
            dualsAtIndex.put(entry.getKey(), atIndexTensor);
        }

        return new DualNumber(DoubleTensor.scalar(this.value.getValue(index)), dualsAtIndex);
    }

    private DoubleTensor takeFromPartial(DoubleTensor from, int... indices) {
        int[] fromShape = from.getShape();
        int[] subFromShape = Arrays.copyOf(fromShape, indices.length);
        int indexToTakeFrom = TensorShape.getFlatIndex(subFromShape, TensorShape.getRowFirstStride(subFromShape), indices);
        int[] takeShape = Arrays.copyOfRange(fromShape, indices.length, fromShape.length);
        int subShapeLength = (int) TensorShape.getLength(subFromShape);

        return from.reshape(subShapeLength, -1)
            .slice(0, indexToTakeFrom)
            .reshape(takeShape);
    }

}
