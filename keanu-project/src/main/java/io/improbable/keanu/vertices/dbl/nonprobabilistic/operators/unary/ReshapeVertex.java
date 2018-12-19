package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.HashMap;
import java.util.Map;

public class ReshapeVertex extends DoubleUnaryOpVertex implements Differentiable {

    private static final String PROPOSED_SHAPE_NAME = "proposedShape";

    public ReshapeVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex,
                         @LoadVertexParam(PROPOSED_SHAPE_NAME) long... proposedShape) {
        super(proposedShape, inputVertex);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.reshape(getShape());
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);

        long[] newPartialShape = TensorShape.concat(
            getShape(),
            derivativeOfParentWithRespectToInputs.getWrtShape(inputVertex.getShape())
        );

        return derivativeOfParentWithRespectToInputs.reshape(newPartialShape);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> reshapedDerivatives = new HashMap<>();

        long[] newPartialShape = TensorShape.concat(
            derivativeOfOutputWithRespectToSelf.getOfShape(getShape()),
            inputVertex.getShape()
        );

        reshapedDerivatives.put(inputVertex, derivativeOfOutputWithRespectToSelf.reshape(newPartialShape));

        return reshapedDerivatives;
    }

    @SaveVertexParam(PROPOSED_SHAPE_NAME)
    public long[] getProposedShape() {
        return getShape();
    }
}
