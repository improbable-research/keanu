package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;
import io.improbable.keanu.vertices.tensor.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.Differentiable;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Collections;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.getReductionResultShape;

public class ProductVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private final static String OVER_DIMENSIONS = "overDimensions";
    private final int[] overDimensions;

    @ExportVertexToPythonBindings
    public ProductVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                         @LoadVertexParam(OVER_DIMENSIONS) int[] overDimensions) {
        super(getReductionResultShape(inputVertex.getShape(), overDimensions), inputVertex, inputVertex.ofType());
        this.overDimensions = overDimensions;
    }

    public ProductVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(new long[0], inputVertex, inputVertex.ofType());
        this.overDimensions = null;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        if (overDimensions == null) {
            return value.product();
        } else {
            return value.product(overDimensions);
        }
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {

        final PartialDerivative partial = derivativeOfParentsWithRespectToInput.get(inputVertex);

        final long[] ofShapeWithoutRankLoss = TensorShape.getReductionResultShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);
        int[] sumOver = overDimensions == null ? TensorShape.dimensionRange(0, inputVertex.getRank()) : overDimensions;

        final DoubleTensor result = partial
            .multiplyAlongOfDimensions(getValue().toDouble().reshape(ofShapeWithoutRankLoss))
            .divideByAlongOfDimensions(inputVertex.getValue().toDouble())
            .get().sum(sumOver);

        return new PartialDerivative(result);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative partial) {

        final long[] wrtShapeWithoutRankLoss = TensorShape.getReductionResultShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);
        final long[] ofShape = partial.getOfShape(this.getShape());

        final long[] newPartialShape = TensorShape.concat(
            ofShape,
            wrtShapeWithoutRankLoss
        );

        final DoubleTensor partialDueToShapeChange = partial.get().reshape(newPartialShape);

        final DoubleTensor result = partialDueToShapeChange
            .times(getValue().toDouble().reshape(wrtShapeWithoutRankLoss))
            .div(inputVertex.getValue().toDouble());

        return Collections.singletonMap(inputVertex, new PartialDerivative(result));
    }

    @SaveVertexParam(OVER_DIMENSIONS)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}
