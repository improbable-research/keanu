package io.improbable.keanu.vertices.intgr;

import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.DiscreteVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.CastIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerDivisionVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerAbsVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary.IntegerUnaryOpLambda;

import java.util.Map;
import java.util.function.Function;

public abstract class IntegerVertex extends DiscreteVertex<IntegerTensor> implements IntegerOperators<IntegerVertex> {

    public IntegerVertex minus(IntegerVertex that) {
        return new IntegerDifferenceVertex(this, that);
    }

    public IntegerVertex plus(IntegerVertex that) {
        return new IntegerAdditionVertex(this, that);
    }

    public IntegerVertex multiply(IntegerVertex that) {
        return new IntegerMultiplicationVertex(this, that);
    }

    public IntegerVertex divideBy(IntegerVertex that) {
        return new IntegerDivisionVertex(this, that);
    }

    public IntegerVertex minus(Vertex<IntegerTensor> that) {
        return new IntegerDifferenceVertex(this, new CastIntegerVertex(that));
    }

    public IntegerVertex plus(Vertex<IntegerTensor> that) {
        return new IntegerAdditionVertex(this, new CastIntegerVertex(that));
    }

    public IntegerVertex multiply(Vertex<IntegerTensor> that) {
        return new IntegerMultiplicationVertex(this, new CastIntegerVertex(that));
    }

    public IntegerVertex divideBy(Vertex<IntegerTensor> that) {
        return new IntegerDivisionVertex(this, new CastIntegerVertex(that));
    }

    public IntegerVertex minus(int value) {
        return new IntegerDifferenceVertex(this, new ConstantIntegerVertex(value));
    }

    public IntegerVertex plus(int value) {
        return new IntegerAdditionVertex(this, new ConstantIntegerVertex(value));
    }

    public IntegerVertex multiply(int factor) {
        return new IntegerMultiplicationVertex(this, new ConstantIntegerVertex(factor));
    }

    public IntegerVertex divideBy(int divisor) {
        return new IntegerDivisionVertex(this, new ConstantIntegerVertex(divisor));
    }

    public IntegerVertex abs() {
        return new IntegerAbsVertex(this);
    }

    public IntegerVertex lambda(int[] shape, Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(shape, this, op);
    }

    public IntegerVertex lambda(Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(this.getShape(), this, op);
    }

    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)
    public IntegerVertex times(IntegerVertex that) {
        return multiply(that);
    }

    public IntegerVertex div(IntegerVertex that) {
        return divideBy(that);
    }

    public IntegerVertex times(int that) {
        return multiply(that);
    }

    public IntegerVertex div(int that) {
        return divideBy(that);
    }

    public IntegerVertex unaryMinus() {
        return multiply(-1);
    }

    public void setValue(int value) {
        super.setValue(IntegerTensor.create(value, getShape()));
    }

    public void setValue(int[] values) {
        super.setValue(IntegerTensor.create(values, getShape()));
    }

    public void setAndCascade(int value) {
        super.setAndCascade(IntegerTensor.create(value, getShape()));
    }

    public void setAndCascade(int[] values) {
        super.setAndCascade(IntegerTensor.create(values, getShape()));
    }

    public void setAndCascade(int value, Map<Long, Long> explored) {
        super.setAndCascade(IntegerTensor.create(value, getShape()), explored);
    }

    public void setAndCascade(int[] values, Map<Long, Long> explored) {
        super.setAndCascade(IntegerTensor.create(values, getShape()), explored);
    }

    public void observe(int value) {
        super.observe(IntegerTensor.create(value, getShape()));
    }

    public void observe(int[] values) {
        super.observe(IntegerTensor.create(values));
    }

    public double logPmf(int value) {
        return this.logPmf(IntegerTensor.scalar(value));
    }

    public double logPmf(int[] values) {
        return this.logPmf(IntegerTensor.create(values));
    }

    public Map<Long, DoubleTensor> dLogPmf(int value) {
        return this.dLogPmf(IntegerTensor.scalar(value));
    }

    public Map<Long, DoubleTensor> dLogPmf(int[] values) {
        return this.dLogPmf(IntegerTensor.create(values));
    }

}
