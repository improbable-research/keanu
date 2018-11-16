package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.vertices.LoadParentVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.DoubleIfVertex;

public class MinVertex extends DoubleIfVertex {

    private static final String LEFT_NAME = "left";
    private static final String RIGHT_NAME = "right";

    /**
     * Finds the minimum between two vertices
     *
     * @param left  one of the vertices to find the minimum of
     * @param right one of the vertices to find the minimum of
     */
    @ExportVertexToPythonBindings
    public MinVertex(@LoadParentVertex(THN_NAME) DoubleVertex left,
                     @LoadParentVertex(ELS_NAME) DoubleVertex right) {
        super(left.getShape(), left.lessThanOrEqualTo(right), left, right);
    }
}
