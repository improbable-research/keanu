package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.TensorShapeValidation;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;

public class AssertVertex extends BoolVertex implements NonProbabilistic<BooleanTensor>, NonSaveableVertex {

    private final Vertex<? extends BooleanTensor> predicate;
    private final BooleanTensor expected;
    private final String errorMessage;

    public static AssertVertex allTrue(Vertex<? extends BooleanTensor> predicate) {
        return new AssertVertex(predicate, BooleanTensor.trues(predicate.getShape()));
    }

    public static AssertVertex allTrue(Vertex<? extends BooleanTensor> predicate, String errorMessage) {
        return new AssertVertex(predicate, BooleanTensor.trues(predicate.getShape()), errorMessage);
    }

    public static AssertVertex allFalse(Vertex<? extends BooleanTensor> predicate) {
        return new AssertVertex(predicate, BooleanTensor.falses(predicate.getShape()));
    }

    public static AssertVertex allFalse(Vertex<? extends BooleanTensor> predicate, String errorMessage) {
        return new AssertVertex(predicate, BooleanTensor.falses(predicate.getShape()), errorMessage);
    }

    public AssertVertex(Vertex<? extends BooleanTensor> predicate, BooleanTensor expected,
                        String errorMessage) {
        super(TensorShapeValidation.checkAllShapesMatch(predicate.getShape(), expected.getShape()));
        this.predicate = predicate;
        this.expected = expected;
        this.errorMessage = errorMessage;
        setParents(predicate);
    }

    public AssertVertex(Vertex<? extends BooleanTensor> predicate, BooleanTensor expected) {
        this(predicate, expected, "Failed assertion");
    }


    @Override
    public BooleanTensor calculate() {
        assertion();
        return predicate.getValue();
    }

    private void assertion() {
        if (!predicateMatchesExpected()) {
            throw new AssertionError(errorMessage);
        }
    }

    private boolean predicateMatchesExpected() {
        return predicate.getValue().xor(expected).allFalse();
    }


    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return predicate.sample();
    }

}
