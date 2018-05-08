package io.improbable.keanu.vertices.dbltensor.nonprobabilistic.diff;

import io.improbable.keanu.vertices.dbltensor.DoubleTensor;

import java.util.Collections;
import java.util.Map;

public class TensorDualNumber {

    public static TensorDualNumber createConstant(DoubleTensor value) {
        return new TensorDualNumber(value, TensorPartialDerivatives.OF_CONSTANT);
    }

    public static TensorDualNumber createWithRespectToSelf(String withRespectTo, DoubleTensor value) {
        return new TensorDualNumber(value, TensorPartialDerivatives.withRespectToSelf(withRespectTo, value.getShape()));
    }

    private DoubleTensor value;
    private TensorPartialDerivatives partialDerivatives;

    public TensorDualNumber(DoubleTensor value, TensorPartialDerivatives partialDerivatives) {
        this.value = value;
        this.partialDerivatives = partialDerivatives;
    }

    public TensorDualNumber(DoubleTensor value, Map<String, DoubleTensor> partialDerivatives) {
        this(value, new TensorPartialDerivatives(partialDerivatives));
    }

    public TensorDualNumber(DoubleTensor value, String infinitesimalLabel) {
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
}
