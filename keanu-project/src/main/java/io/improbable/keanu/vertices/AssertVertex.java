package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;


public class AssertVertex extends BoolVertex implements NonProbabilistic<BooleanTensor>, NonSaveableVertex {

    private static final String PREDICATE_NAME = "predicate";
    private static final String ERROR_MESSAGE_NAME = "error";

    private final Vertex<? extends BooleanTensor> predicate;
    private final String errorMessage;

    public AssertVertex(@LoadVertexParam(PREDICATE_NAME) Vertex<? extends BooleanTensor> predicate,
                        @LoadVertexParam(ERROR_MESSAGE_NAME) String errorMessage) {
        super(predicate.getShape());
        this.predicate = predicate;
        this.errorMessage = errorMessage;
        setParents(predicate);
    }

    public AssertVertex(Vertex<? extends BooleanTensor> predicate) {
        this(predicate, "");
    }


    @Override
    public BooleanTensor calculate() {
        assertion();
        return predicate.getValue();
    }

    private void assertion() {
        if (!predicate.getValue().allTrue()) {
            throw new AssertionError(buildAssertMessage());
        }
    }

    private String buildAssertMessage() {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("AssertVertex");
        if (getLabel() != null) {
            stringBuilder.append(" (" + getLabel().getQualifiedName() + ")");
        }
        if (!errorMessage.equals("")) {
            stringBuilder.append(": " + errorMessage);
        }
        return stringBuilder.toString();
    }

    @Override
    public BooleanTensor sample(KeanuRandom random) {
        return predicate.sample();
    }

    @SaveVertexParam(PREDICATE_NAME)
    public Vertex<? extends BooleanTensor> getPredicate() {
        return predicate;
    }

    @SaveVertexParam(ERROR_MESSAGE_NAME)
    public String getErrorMessage() {
        return errorMessage;
    }
}
