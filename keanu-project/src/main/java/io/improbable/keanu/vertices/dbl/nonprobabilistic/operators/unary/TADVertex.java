package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;

import static io.improbable.keanu.tensor.TensorShapeValidation.shapeAlongDimension;

public class TADVertex extends DoubleUnaryOpVertex {

    private final int dimension;
    private final int index;

    /**
     * Takes the tensor along a dimension for a vertex
     *
     * @param inputVertex the input vertex
     * @param dimension the dimension to extract along
     * @param index the index of extraction
     */
    public TADVertex(DoubleVertex inputVertex, int dimension, int index) {
        super(shapeAlongDimension(dimension, inputVertex.getShape()), inputVertex);
        this.dimension = dimension;
        this.index = index;
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        return dualNumbers.get(inputVertex).tad(dimension, index);
    }

    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return a.tad(dimension, index);
    }
}
