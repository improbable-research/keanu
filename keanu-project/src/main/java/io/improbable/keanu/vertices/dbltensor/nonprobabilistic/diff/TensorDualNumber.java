package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

import java.util.Collections;
import java.util.Map;

public class TensorDualNumber {

    public static TensorDualNumber createConstant(DoubleTensor value) {
        return new TensorDualNumber(value, TensorPartialDerivatives.OF_CONSTANT);
    }

    public static TensorDualNumber createWithRespectToSelf(long withRespectTo, DoubleTensor value) {
        return new TensorDualNumber(value, TensorPartialDerivatives.withRespectToSelf(withRespectTo, value.getShape()));
    }

    private DoubleTensor value;
    private TensorPartialDerivatives partialDerivatives;

    public TensorDualNumber(DoubleTensor value, TensorPartialDerivatives partialDerivatives) {
        this.value = value;
        this.partialDerivatives = partialDerivatives;
    }

    public TensorDualNumber(DoubleTensor value, Map<Long, DoubleTensor> partialDerivatives) {
        this(value, new TensorPartialDerivatives(partialDerivatives));
    }

    public TensorDualNumber(DoubleTensor value, long infinitesimalLabel) {
        this(value, new TensorPartialDerivatives(Collections.singletonMap(infinitesimalLabel, DoubleTensor.ones(value.getShape()))));
    }

    public DoubleTensor getValue() {
        return value;
    }

    public TensorPartialDerivatives getPartialDerivatives() {
        return partialDerivatives;
    }

    public boolean isOfConstant() {
        return partialDerivatives.isEmpty();
    }

    public TensorDualNumber add(TensorDualNumber that) {
        // dc = da + db;
        DoubleTensor newValue = this.value.plus(that.value);
        TensorPartialDerivatives newInf = this.partialDerivatives.add(that.partialDerivatives);
        return new TensorDualNumber(newValue, newInf);
    }

    public TensorDualNumber subtract(TensorDualNumber that) {
        // dc = da - db;
        DoubleTensor newValue = this.value.minus(that.value);
        TensorPartialDerivatives newInf = this.partialDerivatives.subtract(that.partialDerivatives);
        return new TensorDualNumber(newValue, newInf);
    }

    public TensorDualNumber multiplyBy(TensorDualNumber that) {
        // dc = A * db + B * da;
        DoubleTensor newValue = this.value.times(that.value);
        TensorPartialDerivatives thisInfMultiplied = this.partialDerivatives.multiplyBy(that.value);
        TensorPartialDerivatives thatInfMultiplied = that.partialDerivatives.multiplyBy(this.value);
        TensorPartialDerivatives newInf = thisInfMultiplied.add(thatInfMultiplied);
        return new TensorDualNumber(newValue, newInf);
    }

    public TensorDualNumber divideBy(TensorDualNumber that) {
        // dc = (B * da - A * db) / B^2;
        DoubleTensor newValue = this.value.div(that.value);
        TensorPartialDerivatives thisInfMultiplied = this.partialDerivatives.multiplyBy(that.value);
        TensorPartialDerivatives thatInfMultiplied = that.partialDerivatives.multiplyBy(this.value);
        TensorPartialDerivatives newInf = thisInfMultiplied.subtract(thatInfMultiplied).divideBy(that.value.times(that.value));
        return new TensorDualNumber(newValue, newInf);
    }

    public TensorDualNumber pow(TensorDualNumber that) {
        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        DoubleTensor newValue = this.value.pow(that.value);
        TensorPartialDerivatives thisInfBase = this.partialDerivatives.multiplyBy(that.value.times(this.value.pow(that.value.minus(1))));
        TensorPartialDerivatives thisInfExponent = that.partialDerivatives.multiplyBy(this.value.log().timesInPlace(newValue));
        TensorPartialDerivatives newInf = thisInfBase.add(thisInfExponent);
        return new TensorDualNumber(newValue, newInf);
    }

    public TensorDualNumber plus(TensorDualNumber that) {
        return add(that);
    }

    public TensorDualNumber minus(TensorDualNumber that) {
        return subtract(that);
    }

    public TensorDualNumber times(TensorDualNumber that) {
        return multiplyBy(that);
    }

    public TensorDualNumber div(TensorDualNumber that) {
        return divideBy(that);
    }

    public TensorDualNumber plus(double value) {
        DoubleTensor newValue = this.value.plus(value);
        TensorPartialDerivatives clonedInf = this.partialDerivatives.clone();
        return new TensorDualNumber(newValue, clonedInf);
    }

    public TensorDualNumber minus(double value) {
        DoubleTensor newValue = this.value.minus(value);
        TensorPartialDerivatives clonedInf = this.partialDerivatives.clone();
        return new TensorDualNumber(newValue, clonedInf);
    }

    public TensorDualNumber times(double value) {
        DoubleTensor newValue = this.value.times(value);
        TensorPartialDerivatives newInf = this.partialDerivatives.multiplyBy(value);
        return new TensorDualNumber(newValue, newInf);
    }

    public TensorDualNumber div(double value) {
        DoubleTensor newValue = this.value.div(value);
        TensorPartialDerivatives newInf = this.partialDerivatives.divideBy(value);
        return new TensorDualNumber(newValue, newInf);
    }

    public TensorDualNumber unaryMinus() {
        return times(-1.0);
    }

    public TensorDualNumber exp() {
        DoubleTensor eVal = value.exp();
        return new TensorDualNumber(eVal, getPartialDerivatives().multiplyBy(eVal));
    }

    public TensorDualNumber sin() {
        return new TensorDualNumber(value.sin(), getPartialDerivatives().multiplyBy(value.cos()));
    }

    public TensorDualNumber cos() {
        return new TensorDualNumber(value.cos(), getPartialDerivatives().multiplyBy(value.sin().unaryMinusInPlace()));
    }

    public TensorDualNumber tan() {
        DoubleTensor dTan = value.cos().powInPlace(2).reciprocalInPlace();
        return new TensorDualNumber(value.tan(), getPartialDerivatives().multiplyBy(dTan));
    }

    public TensorDualNumber asin() {
        DoubleTensor dArcSin = (value.unaryMinus().timesInPlace(value).plusInPlace(1)).sqrtInPlace().reciprocalInPlace();
        return new TensorDualNumber(value.sin(), getPartialDerivatives().multiplyBy(dArcSin));
    }

    public TensorDualNumber acos() {
        DoubleTensor dArcCos = value.unaryMinus().timesInPlace(value).plusInPlace(1).sqrtInPlace().reciprocalInPlace().unaryMinusInPlace();
        return new TensorDualNumber(value.acos(), getPartialDerivatives().multiplyBy(dArcCos));
    }

    public TensorDualNumber atan() {
        DoubleTensor dArcTan = value.powInPlace(2).plusInPlace(1).reciprocalInPlace();
        return new TensorDualNumber(value.atan(), getPartialDerivatives().multiplyBy(dArcTan));
    }

    public TensorDualNumber log() {
        return new TensorDualNumber(value.log(), getPartialDerivatives().divideBy(value));
    }

}
