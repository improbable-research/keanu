package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.Collections;
import java.util.Map;


public class DualNumber {

    private DoubleTensor value;
    private PartialDerivatives partialDerivatives;

    public DualNumber(DoubleTensor value, PartialDerivatives partialDerivatives) {
        this.value = value;
        this.partialDerivatives = partialDerivatives;
    }

    public DualNumber(DoubleTensor value, Map<String, DoubleTensor> partialDerivatives) {
        this(value, new PartialDerivatives(partialDerivatives));
    }

    public DualNumber(DoubleTensor value, String infinitesimalLabel) {
        this(value, new PartialDerivatives(Collections.singletonMap(infinitesimalLabel, DoubleTensor.ones(value.getShape()))));
    }

    public DoubleTensor getValue() {
        return value;
    }

    public PartialDerivatives getPartialDerivatives() {
        return partialDerivatives;
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
}
