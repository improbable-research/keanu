package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
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
    public MinVertex(@LoadVertexParam(LEFT_NAME) DoubleVertex left,
                     @LoadVertexParam(RIGHT_NAME) DoubleVertex right) {
        super(left.lessThanOrEqual(right), left, right);
    }
}
