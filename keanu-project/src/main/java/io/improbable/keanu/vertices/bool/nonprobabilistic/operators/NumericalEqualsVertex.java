package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.NonProbabilisticBool;

import java.util.Random;

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
    public Boolean sample(Random random) {
        return op(a.sample(random), b.sample(random), epsilon.sample(random));
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
