package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.kotlin.DoubleOperators;


public class DualNumber implements DoubleOperators<DualNumber> {

    public static DualNumber createConstant(double value) {
        return new DualNumber(value, PartialDerivatives.OF_CONSTANT);
    }

    public static DualNumber createWithRespectToSelf(String withRespectTo, double value) {
        return new DualNumber(value, PartialDerivatives.withRespectToSelf(withRespectTo));
    }

    private double value;
    private PartialDerivatives partialDerivatives;

    public DualNumber(double value, PartialDerivatives partialDerivatives) {
        this.value = value;
        this.partialDerivatives = partialDerivatives;
    }

    public double getValue() {
        return value;
    }

    public PartialDerivatives getPartialDerivatives() {
        return partialDerivatives;
    }

    public boolean isOfConstant(){
        return partialDerivatives.isEmpty();
    }

    public DualNumber add(DualNumber that) {
        // dc = da + db;
        double newValue = this.value + that.value;
        PartialDerivatives newInf = this.partialDerivatives.add(that.partialDerivatives);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber subtract(DualNumber that) {
        // dc = da - db;
        double newValue = this.value - that.value;
        PartialDerivatives newInf = this.partialDerivatives.subtract(that.partialDerivatives);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber multiplyBy(DualNumber that) {
        // dc = A * db + B * da;
        double newValue = this.value * that.value;
        PartialDerivatives thisInfMultiplied = this.partialDerivatives.multiplyBy(that.value);
        PartialDerivatives thatInfMultiplied = that.partialDerivatives.multiplyBy(this.value);
        PartialDerivatives newInf = thisInfMultiplied.add(thatInfMultiplied);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber divideBy(DualNumber that) {
        // dc = (B * da - A * db) / B^2;
        double newValue = this.value / that.value;
        PartialDerivatives thisInfMultiplied = this.partialDerivatives.multiplyBy(that.value);
        PartialDerivatives thatInfMultiplied = that.partialDerivatives.multiplyBy(this.value);
        PartialDerivatives newInf = thisInfMultiplied.subtract(thatInfMultiplied).divideBy(that.value * that.value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber pow(DualNumber that) {
        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        double newValue = Math.pow(this.value, that.value);
        PartialDerivatives thisInfBase = this.partialDerivatives.multiplyBy(that.value * Math.pow(this.value, that.value - 1));
        PartialDerivatives thisInfExponent = that.partialDerivatives.multiplyBy(Math.log(this.value) * newValue);
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
        double newValue = this.value + value;
        PartialDerivatives clonedInf = this.partialDerivatives.copy();
        return new DualNumber(newValue, clonedInf);
    }

    public DualNumber minus(double value) {
        double newValue = this.value - value;
        PartialDerivatives clonedInf = this.partialDerivatives.copy();
        return new DualNumber(newValue, clonedInf);
    }

    public DualNumber times(double value) {
        double newValue = this.value * value;
        PartialDerivatives newInf = this.partialDerivatives.multiplyBy(value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber div(double value) {
        double newValue = this.value / value;
        PartialDerivatives newInf = this.partialDerivatives.divideBy(value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber pow(double value) {
        double newValue = Math.pow(this.value, value);
        PartialDerivatives newInf = this.partialDerivatives.powerTo(value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber unaryMinus() {
        return times(-1.0);
    }

    public DualNumber log() {
        return new DualNumber(Math.log(getValue()), getPartialDerivatives().divideBy(getValue()));
    }

    public DualNumber exp() {
        double eVal = Math.exp(getValue());
        return new DualNumber(eVal, getPartialDerivatives().multiplyBy(eVal));
    }

    public DualNumber sin() {
        return new DualNumber(Math.sin(getValue()), getPartialDerivatives().multiplyBy(Math.cos(getValue())));
    }

    public DualNumber cos() {
        return new DualNumber(Math.cos(getValue()), getPartialDerivatives().multiplyBy(-Math.sin(getValue())));
    }

    public DualNumber asin() {
        double dArcSin = 1.0 / Math.sqrt(1.0 - getValue() * getValue());
        return new DualNumber(Math.asin(getValue()), getPartialDerivatives().multiplyBy(dArcSin));
    }

    public DualNumber acos() {
        double dArcCos = -1.0 / Math.sqrt(1.0 - getValue() * getValue());
        return new DualNumber(Math.acos(getValue()), getPartialDerivatives().multiplyBy(dArcCos));
    }
}
