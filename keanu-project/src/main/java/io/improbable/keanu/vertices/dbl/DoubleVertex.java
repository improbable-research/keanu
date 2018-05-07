package io.improbable.keanu.vertices.dbl;


import io.improbable.keanu.kotlin.DoubleOperators;
import io.improbable.keanu.vertices.ContinuousVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.CastDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.*;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary.*;

import java.util.*;
import java.util.function.Function;

public abstract class DoubleVertex extends ContinuousVertex<Double> implements DoubleOperators<DoubleVertex> {

    /**
     * Calculate the Dual Number of a DoubleVertex.
     *
     * @param dualNumbers A Map that is guaranteed to contain the Dual Numbers of the parent of the vertex.
     * @return The Dual Number of the vertex.
     */
    protected abstract DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers);

    public DualNumber getDualNumber() {
        Map<Vertex, DualNumber> dualNumbers = new HashMap<>();
        Deque<DoubleVertex> stack = new ArrayDeque<>();
        stack.push(this);

        while (!stack.isEmpty()) {

            DoubleVertex head = stack.peek();
            Set<Vertex> parentsThatDualNumberIsNotCalculated = parentsThatDualNumberIsNotCalculated(dualNumbers, head.getParents());

            if (parentsThatDualNumberIsNotCalculated.isEmpty()) {

                DoubleVertex top = stack.pop();
                DualNumber dual = top.calculateDualNumber(dualNumbers);
                dualNumbers.put(top, dual);

            } else {

                for (Vertex<?> vertex : parentsThatDualNumberIsNotCalculated) {
                    if (vertex instanceof DoubleVertex) {
                        stack.push((DoubleVertex) vertex);
                    } else {
                        throw new IllegalArgumentException("Can only calculate Dual Numbers on a graph made of Doubles");
                    }
                }

            }

        }
        return dualNumbers.get(this);
    }

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

    public DoubleVertex lambda(Function<Double, Double> op, Function<Map<Vertex, DualNumber>, DualNumber> dualNumberCalculation) {
        return new DoubleUnaryOpLambda<>(this, op, dualNumberCalculation);
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

    private Set<Vertex> parentsThatDualNumberIsNotCalculated(Map<Vertex, DualNumber> dualNumbers, Set<Vertex> parents) {
        Set<Vertex> notCalculatedParents = new HashSet<>();
        for (Vertex<?> next : parents) {
            if (!dualNumbers.containsKey(next)){
                notCalculatedParents.add(next);
            }
        }
        return notCalculatedParents;
    }

}
