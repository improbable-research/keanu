package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;

public class MaxVertex extends DoubleIfVertex {

    private static final String LEFT_NAME = THEN_NAME;
    private static final String RIGHT_NAME = ELSE_NAME;

    /**
     * Finds the maximum between two vertices
     *
     * @param left  one of the vertices to find the maximum of
     * @param right one of the vertices to find the maximum of
     */
    @ExportVertexToPythonBindings
    public MaxVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                     @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(left.greaterThanOrEqual(right), left, right);
    }
}
