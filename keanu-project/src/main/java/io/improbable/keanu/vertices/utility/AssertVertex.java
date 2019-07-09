package io.improbable.keanu.vertices.utility;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilistic;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexImpl;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.bool.BooleanVertex;


public class AssertVertex extends VertexImpl<BooleanTensor> implements BooleanVertex, NonProbabilistic<BooleanTensor> {

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
        VertexLabel label = getLabel();
        return assertion(predicate.getValue(), errorMessage, label != null ? label.getQualifiedName() : null);
    }

    public static BooleanTensor assertion(BooleanTensor predicateValue, String errorMessage, String labelQualifiedName) {
        if (!predicateValue.allTrue()) {
            throw new GraphAssertionException(buildAssertMessage(errorMessage, labelQualifiedName));
        }
        return predicateValue;
    }

    private static String buildAssertMessage(String errorMessage, String labelQualifiedName) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append("AssertVertex");
        if (labelQualifiedName != null) {
            stringBuilder.append(" (" + labelQualifiedName + ")");
        }
        if (!errorMessage.equals("")) {
            stringBuilder.append(": " + errorMessage);
        }
        return stringBuilder.toString();
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
