package io.improbable.keanu.vertices.intgrtensor;


import io.improbable.keanu.kotlin.IntegerOperators;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.DiscreteTensorVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.CastIntegerVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.ConstantIntegerVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.binary.IntegerAdditionVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.binary.IntegerDifferenceVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.binary.IntegerDivisionVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.binary.IntegerMultiplicationVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.unary.IntegerAbsVertex;
import io.improbable.keanu.vertices.intgrtensor.nonprobabilistic.operators.unary.IntegerUnaryOpLambda;

import java.util.Map;
import java.util.function.Function;

public abstract class IntegerVertex extends DiscreteTensorVertex<IntegerTensor> implements IntegerOperators<io.improbable.keanu.vertices.intgrtensor.IntegerVertex> {

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex minus(io.improbable.keanu.vertices.intgrtensor.IntegerVertex that) {
        return new IntegerDifferenceVertex(this, that);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex plus(io.improbable.keanu.vertices.intgrtensor.IntegerVertex that) {
        return new IntegerAdditionVertex(this, that);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex multiply(io.improbable.keanu.vertices.intgrtensor.IntegerVertex that) {
        return new IntegerMultiplicationVertex(this, that);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex divideBy(io.improbable.keanu.vertices.intgrtensor.IntegerVertex that) {
        return new IntegerDivisionVertex(this, that);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex minus(Vertex<IntegerTensor> that) {
        return new IntegerDifferenceVertex(this, new CastIntegerVertex(that));
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex plus(Vertex<IntegerTensor> that) {
        return new IntegerAdditionVertex(this, new CastIntegerVertex(that));
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex multiply(Vertex<IntegerTensor> that) {
        return new IntegerMultiplicationVertex(this, new CastIntegerVertex(that));
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex divideBy(Vertex<IntegerTensor> that) {
        return new IntegerDivisionVertex(this, new CastIntegerVertex(that));
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex minus(int value) {
        return new IntegerDifferenceVertex(this, new ConstantIntegerVertex(value));
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex plus(int value) {
        return new IntegerAdditionVertex(this, new ConstantIntegerVertex(value));
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex multiply(int factor) {
        return new IntegerMultiplicationVertex(this, new ConstantIntegerVertex(factor));
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex divideBy(int divisor) {
        return new IntegerDivisionVertex(this, new ConstantIntegerVertex(divisor));
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex abs() {
        return new IntegerAbsVertex(this);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex lambda(Function<IntegerTensor, IntegerTensor> op) {
        return new IntegerUnaryOpLambda<>(this, op);
    }


    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)
    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex times(io.improbable.keanu.vertices.intgrtensor.IntegerVertex that) {
        return multiply(that);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex div(io.improbable.keanu.vertices.intgrtensor.IntegerVertex that) {
        return divideBy(that);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex times(int that) {
        return multiply(that);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex div(int that) {
        return divideBy(that);
    }

    public io.improbable.keanu.vertices.intgrtensor.IntegerVertex unaryMinus() {
        return multiply(-1);
    }

    public void setValue(int value) {
        super.setValue(IntegerTensor.scalar(value));
    }

    public void setAndCascade(int value) {
        super.setAndCascade(IntegerTensor.scalar(value));
    }

    public void setAndCascade(int value, Map<Long, Long> explored) {
        super.setAndCascade(IntegerTensor.scalar(value), explored);
    }

    public void observe(int value) {
        super.observe(IntegerTensor.scalar(value));
    }

    public double logPmf(int value) {
        if (this.getValue().isScalar()) {
            return this.logPmf(IntegerTensor.scalar(value));
        } else {
            throw new IllegalArgumentException("Vertex is not scalar");
        }
    }

    public Map<Long, DoubleTensor> dLogPmf(int value) {
        if (this.getValue().isScalar()) {
            return this.dLogPmf(IntegerTensor.scalar(value));
        } else {
            throw new IllegalArgumentException("Vertex is not scalar");
        }
    }

}
