package io.improbable.keanu.vertices.number.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.NumberTensor;
import io.improbable.keanu.tensor.Tensor;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.NonProbabilisticVertex;
import io.improbable.keanu.vertices.SaveVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;
import io.improbable.keanu.vertices.number.NumberTensorVertex;
import io.improbable.keanu.vertices.tensor.TensorVertex;
import io.improbable.keanu.vertices.tensor.UnaryTensorOpVertex;

import java.util.Arrays;
import java.util.Map;

import static io.improbable.keanu.tensor.TensorShape.getReductionResultShape;
import static java.util.Collections.singletonMap;

public class SumVertex<T extends Number, TENSOR extends NumberTensor<T, TENSOR>, VERTEX extends NumberTensorVertex<T, TENSOR, VERTEX>>
    extends UnaryTensorOpVertex<T, TENSOR, VERTEX> implements NonProbabilisticVertex<TENSOR, VERTEX>, Differentiable {

    private static final String DIMENSIONS_NAME = "overDimensions";
    private final int[] overDimensions;

    /**
     * Performs a sum across specified dimensions. Negative dimension indexing is not supported.
     *
     * @param inputVertex    the vertex to have its values summed
     * @param overDimensions dimensions to sum over
     */
    @ExportVertexToPythonBindings
    public SumVertex(@LoadVertexParam(INPUT_NAME) TensorVertex<T, TENSOR, VERTEX> inputVertex,
                     @LoadVertexParam(DIMENSIONS_NAME) int[] overDimensions) {
        super(getReductionResultShape(inputVertex.getShape(), overDimensions), inputVertex, inputVertex.ofType());
        this.overDimensions = overDimensions;
    }

    /**
     * Performs a sum across all dimensions
     *
     * @param inputVertex the vertex to have its values summed
     */
    public SumVertex(TensorVertex<T, TENSOR, VERTEX> inputVertex) {
        super(Tensor.SCALAR_SHAPE, inputVertex, inputVertex.ofType());
        this.overDimensions = null;
    }

    @Override
    protected TENSOR op(TENSOR value) {
        if (overDimensions == null) {
            return value.sum();
        } else {
            return value.sum(overDimensions);
        }
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative dInputVertex = derivativeOfParentsWithRespectToInput.get(inputVertex);
        int operandRank = inputVertex.getValue().getRank();
        int[] dimensionsToSum = overDimensions == null ? TensorShape.dimensionRange(0, operandRank) : overDimensions;
        return new PartialDerivative(dInputVertex.get().sum(dimensionsToSum));
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {

        long[] wrtShapeWithoutRankLoss = summedOverShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);
        long[] ofShape = derivativeOfOutputWithRespectToSelf.getOfShape(this.getShape());

        long[] newPartialShape = TensorShape.concat(
            ofShape,
            wrtShapeWithoutRankLoss
        );

        DoubleTensor partialDueToSummationShapeChange = derivativeOfOutputWithRespectToSelf.get().reshape(newPartialShape);

        long[] resultShape = TensorShape.concat(
            ofShape,
            inputVertex.getShape()
        );

        DoubleTensor broadcastedPartial = partialDueToSummationShapeChange.broadcast(resultShape);

        return singletonMap(inputVertex, new PartialDerivative(broadcastedPartial));
    }

    private static long[] summedOverShapeWithoutRankLoss(long[] shape, int[] sumOverDimensions) {
        long[] shapeCopy = Arrays.copyOf(shape, shape.length);

        if (sumOverDimensions == null) {
            Arrays.fill(shapeCopy, 1L);
        } else {
            for (int sumOverDimension : sumOverDimensions) {
                shapeCopy[sumOverDimension] = 1L;
            }
        }

        return shapeCopy;
    }

    @SaveVertexParam(DIMENSIONS_NAME)
    public int[] getOverDimensions() {
        return overDimensions;
    }
}