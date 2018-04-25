package io.improbable.keanu.vertices.dbl.nonprobabilistic.diff;

import io.improbable.keanu.kotlin.DoubleOperators;

import java.util.Collections;
import java.util.Map;


public class DualNumber implements DoubleOperators<DualNumber> {

    private double value;
    private Infinitesimal infinitesimal;

    public DualNumber(double value, Infinitesimal infinitesimal) {
        this.value = value;
        this.infinitesimal = infinitesimal;
    }

    public DualNumber(double value, Map<String, Double> infinitesimal) {
        this(value, new Infinitesimal(infinitesimal));
    }

    public DualNumber(double value, String infinitesimalLabel) {
        this(value, new Infinitesimal(Collections.singletonMap(infinitesimalLabel, 1.0)));
    }

    public double getValue() {
        return value;
    }

    public Infinitesimal getInfinitesimal() {
        return infinitesimal;
    }

    public DualNumber add(DualNumber that) {
        // dc = da + db;
        double newValue = this.value + that.value;
        Infinitesimal newInf = this.infinitesimal.add(that.infinitesimal);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber subtract(DualNumber that) {
        // dc = da - db;
        double newValue = this.value - that.value;
        Infinitesimal newInf = this.infinitesimal.subtract(that.infinitesimal);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber multiplyBy(DualNumber that) {
        // dc = A * db + B * da;
        double newValue = this.value * that.value;
        Infinitesimal thisInfMultiplied = this.infinitesimal.multiplyBy(that.value);
        Infinitesimal thatInfMultiplied = that.infinitesimal.multiplyBy(this.value);
        Infinitesimal newInf = thisInfMultiplied.add(thatInfMultiplied);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber divideBy(DualNumber that) {
        // dc = (B * da - A * db) / B^2;
        double newValue = this.value / that.value;
        Infinitesimal thisInfMultiplied = this.infinitesimal.multiplyBy(that.value);
        Infinitesimal thatInfMultiplied = that.infinitesimal.multiplyBy(this.value);
        Infinitesimal newInf = thisInfMultiplied.subtract(thatInfMultiplied).divideBy(that.value * that.value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber pow(DualNumber that) {
        // dc = (A ^ B) * B * (dA / A) + (dB * log (A))
        double newValue = Math.pow(this.value, that.value);
        Infinitesimal thisInfBase = this.infinitesimal.multiplyBy(that.value * Math.pow(this.value, that.value - 1));
        Infinitesimal thisInfExponent = that.infinitesimal.multiplyBy(Math.log(this.value) * newValue);
        Infinitesimal newInf = thisInfBase.add(thisInfExponent);
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
        Infinitesimal clonedInf = this.infinitesimal.clone();
        return new DualNumber(newValue, clonedInf);
    }

    public DualNumber minus(double value) {
        double newValue = this.value - value;
        Infinitesimal clonedInf = this.infinitesimal.clone();
        return new DualNumber(newValue, clonedInf);
    }

    public DualNumber times(double value) {
        double newValue = this.value * value;
        Infinitesimal newInf = this.infinitesimal.multiplyBy(value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber div(double value) {
        double newValue = this.value / value;
        Infinitesimal newInf = this.infinitesimal.divideBy(value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber pow(double value) {
        double newValue = Math.pow(this.value, value);
        Infinitesimal newInf = this.infinitesimal.powerTo(value);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber unaryMinus() {
        return times(-1.0);
    }

    public DualNumber log() {
        return new DualNumber(Math.log(getValue()), getInfinitesimal().divideBy(getValue()));
    }

    public DualNumber exp() {
        double eVal = Math.exp(getValue());
        return new DualNumber(eVal, getInfinitesimal().multiplyBy(eVal));
    }

    public DualNumber sin() {
        return new DualNumber(Math.sin(getValue()), getInfinitesimal().multiplyBy(Math.cos(getValue())));
    }

    public DualNumber cos() {
        return new DualNumber(Math.cos(getValue()), getInfinitesimal().multiplyBy(-Math.sin(getValue())));
    }

    public DualNumber asin() {
        double dArcSin = 1.0 / Math.sqrt(1.0 - getValue() * getValue());
        return new DualNumber(Math.asin(getValue()), getInfinitesimal().multiplyBy(dArcSin));
    }

    public DualNumber acos() {
        double dArcCos = -1.0 / Math.sqrt(1.0 - getValue() * getValue());
        return new DualNumber(Math.acos(getValue()), getInfinitesimal().multiplyBy(dArcCos));
    }
}
