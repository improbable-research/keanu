package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.operators.binary.compare.LessThanOrEqualVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;

public class MinVertex extends DoubleIfVertex {

    private final static String LEFT_NAME = THEN_NAME;
    private final static String RIGHT_NAME = ELSE_NAME;

    /**
     * Finds the minimum between two vertices
     *
     * @param left  one of the vertices to find the minimum of
     * @param right one of the vertices to find the minimum of
     */
    @ExportVertexToPythonBindings
    public MinVertex(@LoadVertexParam(LEFT_NAME) Vertex<DoubleTensor, ?> left,
                     @LoadVertexParam(RIGHT_NAME) Vertex<DoubleTensor, ?> right) {
        super(new LessThanOrEqualVertex<>(left, right), left, right);
    }
}
