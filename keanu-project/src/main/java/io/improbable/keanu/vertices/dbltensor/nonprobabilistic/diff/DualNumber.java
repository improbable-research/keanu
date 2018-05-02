package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.Collections;
import java.util.Map;


public class DualNumber {

    private DoubleTensor value;
    private Infinitesimal infinitesimal;

    public DualNumber(DoubleTensor value, Infinitesimal infinitesimal) {
        this.value = value;
        this.infinitesimal = infinitesimal;
    }

    public DualNumber(DoubleTensor value, Map<String, DoubleTensor> infinitesimal) {
        this(value, new Infinitesimal(infinitesimal));
    }

    public DualNumber(DoubleTensor value, String infinitesimalLabel) {
        this(value, new Infinitesimal(Collections.singletonMap(infinitesimalLabel, 1.0)));
    }

    public DoubleTensor getValue() {
        return value;
    }

    public Infinitesimal getInfinitesimal() {
        return infinitesimal;
    }

    public DualNumber add(DualNumber that) {
        // dc = da + db;
        DoubleTensor newValue = this.value + that.value;
        Infinitesimal newInf = this.infinitesimal.add(that.infinitesimal);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber subtract(DualNumber that) {
        // dc = da - db;
        DoubleTensor newValue = this.value - that.value;
        Infinitesimal newInf = this.infinitesimal.subtract(that.infinitesimal);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber multiplyBy(DualNumber that) {
        // dc = A * db + B * da;
        DoubleTensor newValue = this.value * that.value;
        Infinitesimal thisInfMultiplied = this.infinitesimal.multiplyBy(that.value);
        Infinitesimal thatInfMultiplied = that.infinitesimal.multiplyBy(this.value);
        Infinitesimal newInf = thisInfMultiplied.add(thatInfMultiplied);
        return new DualNumber(newValue, newInf);
    }

    public DualNumber divideBy(DualNumber that) {
        // dc = (B * da - A * db) / B^2;
        DoubleTensor newValue = this.value / that.value;
        Infinitesimal thisInfMultiplied = this.infinitesimal.multiplyBy(that.value);
        Infinitesimal thatInfMultiplied = that.infinitesimal.multiplyBy(this.value);
        Infinitesimal newInf = thisInfMultiplied.subtract(thatInfMultiplied).divideBy(that.value * that.value);
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

    public DualNumber unaryMinus() {
        return times(-1.0);
    }
}
