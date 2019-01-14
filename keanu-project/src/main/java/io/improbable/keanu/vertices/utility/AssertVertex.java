package io.improbable.keanu.vertices.utility;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.dbl.KeanuRandom;


public class AssertVertex extends BooleanVertex implements NonProbabilistic<BooleanTensor> {

    private static final String PREDICATE_NAME = "predicate";
    private static final String ERROR_MESSAGE_NAME = "error";

    private final Vertex<? extends BooleanTensor> predicate;
    private final String errorMessage;

    /**
     * A vertex that asserts a {@link BooleanVertex} is all true on calculation.
     *
     * @param predicate    the predicate to evaluate
     * @param errorMessage a message to include in the {@link AssertionError}
     * @throws AssertionError if any element of the predicate is false when calculated.
     */
    @ExportVertexToPythonBindings
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
            throw new GraphAssertionException(buildAssertMessage());
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
