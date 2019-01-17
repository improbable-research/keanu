package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang3.ArrayUtils;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

public class PermuteVertex extends DoubleUnaryOpVertex implements Differentiable {

    private final static String REARRANGE_NAME = "rearrange";

    private final int[] rearrange;

    @ExportVertexToPythonBindings
    public PermuteVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex,
                         @LoadVertexParam(REARRANGE_NAME) int... rearrange) {
        super(inputVertex);
        this.rearrange = rearrange;
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.permute(rearrange);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
        int[] range = TensorShape.dimensionRange(0, derivativeOfParentWithRespectToInputs.get().getRank());
        for (int i = 0; i < rearrange.length; i++) {
            range[i] = rearrange[i];
        }
        return derivativeOfParentWithRespectToInputs.permute(range);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        int[] range = TensorShape.dimensionRange(0, derivativeOfOutputWithRespectToSelf.get().getRank());
        int rank = inputVertex.getRank();

        for (int i = 0; i < rank; i++) {
            range[i] = i + rank;
        }

        int[] reversePermute = reversePermute();
        for (int i = rank; i < reversePermute.length + rank; i++) {
            range[i] = reversePermute[i - rank];
        }

        partials.put(inputVertex, derivativeOfOutputWithRespectToSelf.permute(range));
        return partials;
    }

    private int[] reversePermute() {
        int[] reversedPermute = new int[rearrange.length];
        for (int i = 0; i < reversedPermute.length; i++) {
            for (int j = 0; j < rearrange.length; j++) {
                if (i == rearrange[j]) {
                    reversedPermute[i] = j;
                }
            }
        }
        return reversedPermute;
    }
}
