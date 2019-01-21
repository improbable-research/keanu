package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.TensorShape;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.dbl.Differentiable;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.diff.PartialDerivative;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class PermuteVertex extends DoubleUnaryOpVertex implements Differentiable {

    private final static String REARRANGE_NAME = "rearrange";

    private final int[] rearrange;

    @ExportVertexToPythonBindings
    public PermuteVertex(@LoadVertexParam(INPUT_VERTEX_NAME) DoubleVertex inputVertex,
                         @LoadVertexParam(REARRANGE_NAME) int... rearrange) {
        super(DoubleTensor.create(Arrays.stream(inputVertex.getShape()).asDoubleStream().toArray(), inputVertex.getShape()).permute(rearrange).getShape(), inputVertex);
        this.rearrange = rearrange;
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.permute(rearrange);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
        int[] permute = forwardPermute(derivativeOfParentWithRespectToInputs);
        return derivativeOfParentWithRespectToInputs.permute(permute);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        int[] reversePermute = reversePermute(derivativeOfOutputWithRespectToSelf);
        partials.put(inputVertex, derivativeOfOutputWithRespectToSelf.permute(reversePermute));
        return partials;
    }

    private int[] forwardPermute(PartialDerivative partial) {

        int[] permute = new int[partial.get().getRank()];
        for (int i = 0; i < rearrange.length; i++) {
            permute[i] = rearrange[i];
            permute[i + rearrange.length] = i + rearrange.length;
        }
        return permute;
    }

    private int[] reversePermute(PartialDerivative partial) {

        int[] reversedPermute = new int[partial.get().getRank()];
        for (int i = 0; i < rearrange.length; i++) {
            for (int j = 0; j < rearrange.length; j++) {
                if (i == rearrange[j]) {
                    reversedPermute[i] = i;
                    reversedPermute[i + rearrange.length] = j + rearrange.length;
                }
            }
        }
        return reversedPermute;
    }
}
