package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;

public class NumericalEqualsVertex extends NonProbabilisticBool {

    protected Vertex<Number> a;
    protected Vertex<Number> b;
    private Vertex<Number> epsilon;

    /**
     * Returns true if a is within epsilon of b
     */
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
