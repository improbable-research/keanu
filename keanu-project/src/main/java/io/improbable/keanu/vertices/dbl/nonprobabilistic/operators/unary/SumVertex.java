package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import com.google.common.primitives.Longs;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;
import lombok.Getter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.util.Collections.singletonMap;

public class SumVertex extends DoubleUnaryOpVertex {

    @Getter
    private final int[] overDimensions;

    /**
     * Performs a sum across specified dimensions. Negative dimension indexing is not supported.
     *
     * @param inputVertex    the vertex to have its values summed
     * @param overDimensions dimensions to sum over
     */
    public SumVertex(DoubleVertex inputVertex, int[] overDimensions) {
        super(getSummationResultShape(inputVertex.getShape(), overDimensions), inputVertex);
        this.overDimensions = overDimensions;
    }

    /**
     * Performs a sum across all dimensions
     *
     * @param inputVertex the vertex to have its values summed
     */
    public SumVertex(DoubleVertex inputVertex) {
        this(inputVertex, TensorShape.dimensionRange(0, inputVertex.getShape().length));
    }

    private static long[] getSummationResultShape(long[] inputShape, int[] sumOverDimensions) {
        List<Long> inputShapeList = new ArrayList<>(Longs.asList(inputShape));

        zeroOutSummedDimensions(inputShapeList, sumOverDimensions);

        return removeZerosWhenRankGreaterThan2(inputShapeList);
    }

    private static void zeroOutSummedDimensions(List<Long> inputShapeList, int[] sumOverDimensions) {
        for (int dim : sumOverDimensions) {
            inputShapeList.set(dim, 0l);
        }
    }

    /**
     * This is here due to strange behavior in tensor summing over dimensions where
     * dimensions are not dropped if the rank is 2 or less.
     */
    private static long[] removeZerosWhenRankGreaterThan2(List<Long> inputShapeList) {
        for (int i = inputShapeList.size() - 1; i >= 0; i--) {
            if (inputShapeList.get(i) == 0) {
                if (inputShapeList.size() > 2) {
                    inputShapeList.remove(i);
                } else {
                    inputShapeList.set(i, 1l);
                }
            }
        }

        return Longs.toArray(inputShapeList);
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.sum(overDimensions);
    }

    @Override
    protected PartialDerivatives forwardModeAutoDifferentiation(PartialDerivatives derivativeOfParentWithRespectToInputs) {
        DoubleTensor sumResult = this.getValue();
        int operandRank = inputVertex.getValue().getRank();
        return derivativeOfParentWithRespectToInputs.sumOverOfDimensions(overDimensions, sumResult.getShape(), operandRank);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        long[] wrtShapeWithoutRankLoss = summedOverShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);

        PartialDerivatives partialDueToSummationShapeChange = getPartialDueToSummationShapeChange(derivativeOfOutputsWithRespectToSelf);

        PartialDerivatives derivativesWrtInput = partialDueToSummationShapeChange
            .multiplyAlongWrtDimensions(DoubleTensor.ones(inputVertex.getShape()), wrtShapeWithoutRankLoss);

        return singletonMap(inputVertex, derivativesWrtInput);
    }

    private PartialDerivatives getPartialDueToSummationShapeChange(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        long[] wrtShapeWithoutRankLoss = summedOverShapeWithoutRankLoss(inputVertex.getShape(), overDimensions);
        PartialDerivatives reshapedDiffWrtSelf = new PartialDerivatives(new HashMap<>());
        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor partial = partialDerivative.getValue();

            long[] newPartialShape = TensorShape.concat(
                TensorShape.selectDimensions(0, partial.getRank() - getShape().length, partial.getShape()),
                wrtShapeWithoutRankLoss
            );

            DoubleTensor reshapedPartialDerivative = partialDerivative.getValue().reshape(newPartialShape);

            reshapedDiffWrtSelf.putWithRespectTo(partialDerivative.getKey(), reshapedPartialDerivative);
        }

        return reshapedDiffWrtSelf;
    }

    private static long[] summedOverShapeWithoutRankLoss(long[] shape, int[] sumOverDimensions) {
        long[] shapeCopy = Arrays.copyOf(shape, shape.length);
        for (int sumOverDimension : sumOverDimensions) {
            shapeCopy[sumOverDimension] = 1;
        }
        return shapeCopy;
    }
}
