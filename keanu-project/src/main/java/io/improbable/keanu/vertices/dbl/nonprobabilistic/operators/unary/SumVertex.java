package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static java.util.Collections.singletonMap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.google.common.primitives.Ints;

import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexId;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.DualNumber;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivatives;

public class SumVertex extends DoubleUnaryOpVertex {

    private final int[] sumOverDimensions;

    public SumVertex(DoubleVertex inputVertex, int[] sumOverDimensions) {
        super(getSummationResultShape(inputVertex.getShape(), sumOverDimensions), inputVertex);
        this.sumOverDimensions = sumOverDimensions;
    }

    /**
     * This is here due to strange behavior in tensor summing over dimensions where
     * dimensions are not dropped if the rank is 2 or less.
     */
    private static int[] getSummationResultShape(int[] inputShape, int[] sumOverDimensions) {
        List<Integer> inputShapeList = new ArrayList<>(Ints.asList(inputShape));

        for (int dim : sumOverDimensions) {
            inputShapeList.set(dim, 0);
        }

        for (int i = inputShapeList.size() - 1; i >= 0; i--) {
            if (inputShapeList.get(i) == 0) {
                if (inputShapeList.size() > 2) {
                    inputShapeList.remove(i);
                } else {
                    inputShapeList.set(i, 1);
                }
            }
        }

        return Ints.toArray(inputShapeList);
    }

    /**
     * Performs a sum across each value stored in a vertex
     *
     * @param inputVertex the vertex to have its values summed
     */
    public SumVertex(DoubleVertex inputVertex) {
        this(inputVertex, TensorShape.dimensionRange(0, inputVertex.getShape().length));
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.sum(sumOverDimensions);
    }

    @Override
    protected DualNumber dualOp(DualNumber dualNumber) {
        return dualNumber.sum(sumOverDimensions);
    }

    @Override
    public Map<Vertex, PartialDerivatives> reverseModeAutoDifferentiation(PartialDerivatives derivativeOfOutputsWithRespectToSelf) {

        int[] wrtShape = summedOverShape(inputVertex.getShape(), sumOverDimensions);

        PartialDerivatives reshapedDiffWrtSelf = new PartialDerivatives(new HashMap<>());
        for (Map.Entry<VertexId, DoubleTensor> partialDerivative : derivativeOfOutputsWithRespectToSelf.asMap().entrySet()) {
            DoubleTensor partial = partialDerivative.getValue();

            int[] newPartialShape = TensorShape.concat(
                TensorShape.selectDimensions(0, partial.getRank() - getShape().length - 1, partial.getShape()),
                wrtShape
            );

            DoubleTensor reshapedPartialDerivative = partialDerivative.getValue().reshape(newPartialShape);

            reshapedDiffWrtSelf.putWithRespectTo(partialDerivative.getKey(), reshapedPartialDerivative);
        }

        PartialDerivatives derivativesWrtInput = reshapedDiffWrtSelf
            .multiplyAlongWrtDimensions(DoubleTensor.ones(inputVertex.getShape()), wrtShape);

        return singletonMap(inputVertex, derivativesWrtInput);
    }

    private int[] summedOverShape(int[] shape, int[] sumOverDimensions) {
        int[] shapeCopy = Arrays.copyOf(shape, shape.length);
        for (int i = 0; i < sumOverDimensions.length; i++) {
            shapeCopy[sumOverDimensions[i]] = 1;
        }
        return shapeCopy;
    }
}
