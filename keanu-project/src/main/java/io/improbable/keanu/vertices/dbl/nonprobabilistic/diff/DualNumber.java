package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;

import java.util.Collections;
import java.util.Map;

public class DualNumber {

    public static DualNumber createConstant(DoubleTensor value) {
        return new DualNumber(value, PartialDerivatives.OF_CONSTANT);
    }

    public static DualNumber createWithRespectToSelf(long withRespectTo, DoubleTensor value) {
        return new DualNumber(value, PartialDerivatives.withRespectToSelf(withRespectTo, value.getShape()));
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
        PartialDerivatives newInf = this.partialDerivatives.add(that.partialDerivatives);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber subtract(DualNumber that) {
        // dc = da - db;
        DoubleTensor newValue = this.value.minus(that.value);
        PartialDerivatives newInf = this.partialDerivatives.subtract(that.partialDerivatives);
        return new DualNumber(newValue, newInf);
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
            newInf = thisInfMultiplied.subtract(thatInfMultiplied).divideBy(that.value.times(that.value));
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
            DoubleTensor dArcTan = value.powInPlace(2).plusInPlace(1).reciprocalInPlace();
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

}
