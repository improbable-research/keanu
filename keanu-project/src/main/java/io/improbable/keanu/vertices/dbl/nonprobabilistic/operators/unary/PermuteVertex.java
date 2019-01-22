package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import io.improbable.keanu.annotation.ExportVertexToPythonBindings;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.LoadVertexParam;
import io.improbable.keanu.vertices.SaveVertexParam;
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
        super(calculatePermutedShape(inputVertex, rearrange), inputVertex);
        this.rearrange = rearrange;
    }

    @Override
    protected DoubleTensor op(DoubleTensor value) {
        return value.permute(rearrange);
    }

    @Override
    public PartialDerivative forwardModeAutoDifferentiation(Map<Vertex, PartialDerivative> derivativeOfParentsWithRespectToInput) {
        PartialDerivative derivativeOfParentWithRespectToInputs = derivativeOfParentsWithRespectToInput.get(inputVertex);
        int[] permuteToApply = forwardPermute(derivativeOfParentWithRespectToInputs);
        return derivativeOfParentWithRespectToInputs.permute(permuteToApply);
    }

    @Override
    public Map<Vertex, PartialDerivative> reverseModeAutoDifferentiation(PartialDerivative derivativeOfOutputWithRespectToSelf) {
        Map<Vertex, PartialDerivative> partials = new HashMap<>();
        int[] permuteToApply = reversePermute(derivativeOfOutputWithRespectToSelf);
        partials.put(inputVertex, derivativeOfOutputWithRespectToSelf.permute(permuteToApply));
        return partials;
    }

    private int[] forwardPermute(PartialDerivative partial) {
        long[] wrtShape = partial.getWrtShape(inputVertex.getShape());
        int[] permuteToApply = new int[partial.get().getRank()];

        for (int i = 0; i < rearrange.length; i++) {
            permuteToApply[i] = rearrange[i];
        }

        for (int j = 0; j < wrtShape.length; j++) {
            permuteToApply[j + rearrange.length] = j + rearrange.length;
        }

        return permuteToApply;
    }

    private int[] reversePermute(PartialDerivative partial) {
        long[] ofShape = partial.getOfShape(inputVertex.getShape());
        long[] wrtShape = partial.getWrtShape(ofShape);
        int[] reversedPermute = new int[partial.get().getRank()];

        for (int i = 0; i < ofShape.length; i++) {
            reversedPermute[i] = i;
        }

        for (int j = 0; j < wrtShape.length; j++) {
            for (int k = 0; k < wrtShape.length; k++) {
                if (j == rearrange[k]) {
                    reversedPermute[j + ofShape.length] = k + ofShape.length;
                }
            }
        }

        return reversedPermute;
    }

    private static long[] calculatePermutedShape(DoubleVertex inputVertex, int... rearrange) {
        DoubleTensor inputShape = DoubleTensor.create(Arrays.stream(inputVertex.getShape()).asDoubleStream().toArray(), inputVertex.getShape());
        return inputShape.permute(rearrange).getShape();
    }

    @SaveVertexParam(REARRANGE_NAME)
    public int[] getRearrange() {
        return rearrange;
    }
}
