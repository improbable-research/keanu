package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;

import java.util.Map;

public class ReshapeVertex extends DoubleUnaryOpVertex {

    private int[] proposedShape;

    public ReshapeVertex(DoubleVertex inputVertex, int... proposedShape) {
        super(inputVertex.getShape(), inputVertex);
        this.proposedShape = proposedShape;
    }

    /**
     * Returns the supplied vertex with a new shape of the same length
     */
    @Override
    protected DoubleTensor op(DoubleTensor a) {
        return a.reshape(proposedShape);
    }

    @Override
    protected DualNumber calculateDualNumber(Map<Vertex, DualNumber> dualNumbers) {
        Map<Long, DoubleTensor> partialDerivatives = dualNumbers.get(inputVertex).getPartialDerivatives().asMap();
        int[] vertexShape = inputVertex.getShape();

        for (Map.Entry<Long, DoubleTensor> partialDerivative : partialDerivatives.entrySet()) {
            int[] shape = partialDerivative.getValue().getShape();
            int[] wrtShape = extractWrtShape(shape, vertexShape);
            int[] newPartialShape = TensorShape.concat(proposedShape, wrtShape);

            DoubleTensor reshapedPartialDerivative = partialDerivative.getValue().reshape(newPartialShape);
            partialDerivative.setValue(reshapedPartialDerivative);
        }

        return dualNumbers.get(inputVertex);
    }

    private int[] extractWrtShape(int[] partialDerivativeShape, int[] vertexShape) {
        int[] shapeWrt = new int[partialDerivativeShape.length - vertexShape.length];
        for (int i = vertexShape.length; i < partialDerivativeShape.length; i++) {
            shapeWrt[i - vertexShape.length] = partialDerivativeShape[i];
        }
        return shapeWrt;
    }

}
