package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

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

    public DualNumber multiplyBy(DualNumber that) {
        // dc = A * db + B * da;
        DoubleTensor newValue = this.value.times(that.value);
        PartialDerivatives thisInfMultiplied = this.partialDerivatives.multiplyBy(that.value);
        PartialDerivatives thatInfMultiplied = that.partialDerivatives.multiplyBy(this.value);
        PartialDerivatives newInf = thisInfMultiplied.add(thatInfMultiplied);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber divideBy(DualNumber that) {
        // dc = (B * da - A * db) / B^2;
        DoubleTensor newValue = this.value.div(that.value);
        PartialDerivatives thisInfMultiplied = this.partialDerivatives.multiplyBy(that.value);
        PartialDerivatives thatInfMultiplied = that.partialDerivatives.multiplyBy(this.value);
        PartialDerivatives newInf = thisInfMultiplied.subtract(thatInfMultiplied).divideBy(that.value.times(that.value));
        return new DualNumber(newValue, newInf);
    }

    public DualNumber pow(DualNumber that) {
        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        DoubleTensor newValue = this.value.pow(that.value);
        PartialDerivatives thisInfBase = this.partialDerivatives.multiplyBy(that.value.times(this.value.pow(that.value.minus(1))));
        PartialDerivatives thisInfExponent = that.partialDerivatives.multiplyBy(this.value.log().timesInPlace(newValue));
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
        PartialDerivatives newInf = this.partialDerivatives.divideBy(value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber unaryMinus() {
        return times(-1.0);
    }

    public DualNumber exp() {
        DoubleTensor eVal = value.exp();
        return new DualNumber(eVal, getPartialDerivatives().multiplyBy(eVal));
    }

    public DualNumber sin() {
        return new DualNumber(value.sin(), getPartialDerivatives().multiplyBy(value.cos()));
    }

    public DualNumber cos() {
        return new DualNumber(value.cos(), getPartialDerivatives().multiplyBy(value.sin().unaryMinusInPlace()));
    }

    public DualNumber tan() {
        DoubleTensor dTan = value.cos().powInPlace(2).reciprocalInPlace();
        return new DualNumber(value.tan(), getPartialDerivatives().multiplyBy(dTan));
    }

    public DualNumber asin() {
        DoubleTensor dArcSin = (value.unaryMinus().timesInPlace(value).plusInPlace(1)).sqrtInPlace().reciprocalInPlace();
        return new DualNumber(value.asin(), getPartialDerivatives().multiplyBy(dArcSin));
    }

    public DualNumber acos() {
        DoubleTensor dArcCos = value.unaryMinus().timesInPlace(value).plusInPlace(1).sqrtInPlace().reciprocalInPlace().unaryMinusInPlace();
        return new DualNumber(value.acos(), getPartialDerivatives().multiplyBy(dArcCos));
    }

    public DualNumber atan() {
        DoubleTensor dArcTan = value.powInPlace(2).plusInPlace(1).reciprocalInPlace();
        return new DualNumber(value.atan(), getPartialDerivatives().multiplyBy(dArcTan));
    }

    public DualNumber log() {
        return new DualNumber(value.log(), getPartialDerivatives().divideBy(value));
    }

}
