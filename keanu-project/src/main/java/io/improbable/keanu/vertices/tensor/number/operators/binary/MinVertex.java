package io.improbable.keanu.vertices.tensor.number.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.WhereVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;

public class MinVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends TensorVertex<T, TENSOR, VERTEX>>
    extends WhereVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private final static String LEFT_NAME = THEN_NAME;
    private final static String RIGHT_NAME = ELSE_NAME;

    /**
     * Finds the minimum between two vertices
     *
     * @param left  one of the vertices to find the minimum of
     * @param right one of the vertices to find the minimum of
     */
    @ExportVertexToPythonBindings
    public MinVertex(@LoadVertexParam(LEFT_NAME) Vertex<TENSOR, ?> left,
                     @LoadVertexParam(RIGHT_NAME) Vertex<TENSOR, ?> right) {
        super(new LessThanOrEqualVertex<>(left, right), left, right);
    }
}
