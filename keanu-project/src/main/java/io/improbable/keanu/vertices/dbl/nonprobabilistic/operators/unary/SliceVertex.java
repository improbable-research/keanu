package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static io.improbable.keanu.tensor.TensorShape.shapeSlice;

import io.improbable.keanu.vertices.dbl.DoubleVertex;

public class SliceVertex extends DoubleUnaryOpVertex {
    /**
     * Takes the slice along a given dimension and index of a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension   the dimension to extract along
     * @param index       the index of extraction
     */
    public SliceVertex(DoubleVertex inputVertex, int dimension, int index) {
        super(shapeSlice(dimension, inputVertex.getShape()), inputVertex,
            a -> a.slice(dimension, index),
            a -> a.slice(dimension, index)
        );
    }
}
