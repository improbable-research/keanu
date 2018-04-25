package io.improbable.keanu.vertices.dbl;


import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DifferenceVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.DivisionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.MultiplicationVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.*;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.AbsVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.DoubleUnaryOpLambda;

import java.util.function.Function;
import java.util.function.Supplier;

public abstract class DoubleVertex extends Vertex<Double> implements DoubleOperators<DoubleVertex> {

    public abstract DualNumber getDualNumber();

    public DoubleVertex minus(DoubleVertex that) {
        return new DifferenceVertex(this, that);
    }

    public DoubleVertex plus(DoubleVertex that) {
        return new AdditionVertex(this, that);
    }

    public DoubleVertex multiply(DoubleVertex that) {
        return new MultiplicationVertex(this, that);
    }

    public DoubleVertex divideBy(DoubleVertex that) {
        return new DivisionVertex(this, that);
    }

    public DoubleVertex minus(Vertex<Double> that) {
        return new DifferenceVertex(this, new CastDoubleVertex(that));
    }

    public DoubleVertex plus(Vertex<Double> that) {
        return new AdditionVertex(this, new CastDoubleVertex(that));
    }

    public DoubleVertex multiply(Vertex<Double> that) {
        return new MultiplicationVertex(this, new CastDoubleVertex(that));
    }

    public DoubleVertex divideBy(Vertex<Double> that) {
        return new DivisionVertex(this, new CastDoubleVertex(that));
    }

    public DoubleVertex pow(DoubleVertex power) {
        return new PowerVertex(this, power);
    }

    public DoubleVertex minus(double value) {
        return new DifferenceVertex(this, new ConstantDoubleVertex(value));
    }

    public DoubleVertex plus(double value) {
        return new AdditionVertex(this, new ConstantDoubleVertex(value));
    }

    public DoubleVertex multiply(double factor) {
        return new MultiplicationVertex(this, new ConstantDoubleVertex(factor));
    }

    public DoubleVertex divideBy(double divisor) {
        return new DivisionVertex(this, new ConstantDoubleVertex(divisor));
    }

    public DoubleVertex pow(double power) {
        return new PowerVertex(this, power);
    }

    public DoubleVertex abs() {
        return new AbsVertex(this);
    }

    public DoubleVertex lambda(Function<Double, Double> op, Supplier<DualNumber> dualNumberSupplier) {
        return new DoubleUnaryOpLambda<>(this, op, dualNumberSupplier);
    }


    // 'times' and 'div' are required to enable operator overloading in Kotlin (through the DoubleOperators interface)
    public DoubleVertex times(DoubleVertex that) {
        return multiply(that);
    }

    public DoubleVertex div(DoubleVertex that) {
        return divideBy(that);
    }

    public DoubleVertex times(double that) {
        return multiply(that);
    }

    public DoubleVertex div(double that) {
        return divideBy(that);
    }

    public DoubleVertex unaryMinus() {
        return multiply(-1.0);
    }


    public DoubleVertex log() {
        return new LogVertex(this);
    }

    public DoubleVertex exp() {
        return new ExpVertex(this);
    }

    public DoubleVertex sin() {
        return new SinVertex(this);
    }

    public DoubleVertex cos() {
        return new CosVertex(this);
    }

    public DoubleVertex asin() {
        return new ArcSinVertex(this);
    }

    public DoubleVertex acos() {
        return new ArcCosVertex(this);
    }

}
