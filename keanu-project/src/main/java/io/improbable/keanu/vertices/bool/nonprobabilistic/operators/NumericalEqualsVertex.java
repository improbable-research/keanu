package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;

/**
 * Returns true if a vertex value is equal to another vertex value within an epsilon.
 */
public class NumericalEqualsVertex extends NonProbabilisticBool {

    protected Vertex<Number> a;
    protected Vertex<Number> b;
    private Vertex<Number> epsilon;

    public NumericalEqualsVertex(Vertex<Number> a, Vertex<Number> b, Vertex<Number> epsilon) {
        this.a = a;
        this.b = b;
        this.epsilon = epsilon;
        setParents(a, b, epsilon);
    }

    @Override
    public Boolean sample() {
        return op(a.sample(), b.sample(), epsilon.sample());
    }

    @Override
    public Boolean getDerivedValue() {
        return op(a.getValue(), b.getValue(), epsilon.getValue());
    }

    private Boolean op(Number a, Number b, Number epsilon) {
        double aVal = a.doubleValue();
        double bVal = b.doubleValue();
        return Math.abs(aVal - bVal) <= epsilon.doubleValue();
    }

}
