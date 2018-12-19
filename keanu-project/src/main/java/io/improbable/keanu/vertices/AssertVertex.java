package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class AssertVertex extends BoolVertex implements NonProbabilistic<BooleanTensor>, NonSaveableVertex {

    private final Vertex<? extends BooleanTensor> actual;
    private final BooleanTensor expected;

    public AssertVertex(Vertex<? extends BooleanTensor> predicate, BooleanTensor expected) {
        super(TensorShapeValidation.checkAllShapesMatch(predicate.getShape(), expected.getShape()));
        this.actual = predicate;
        this.expected = expected;
        setParents(predicate);
    }

    private boolean actualMatchesExpected() {
        return actual.getValue().xor(expected).allFalse();
    }

    private void assertion() {
        if(!actualMatchesExpected()) {
            throw new AssertionError("Asserted value does not match");
        }
    }

    @Override
    public BooleanTensor calculate() {
        assertion();
        return actual.getValue();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return actual.sample();
    }
}
