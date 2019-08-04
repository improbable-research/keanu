package io.improbable.keanu.vertices.tensor;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.jvm.Slicer;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import io.improbable.keanu.vertices.tensor.number.UnaryOperationTestHelpers;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardModeGradient;
import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesReverseModeGradient;

public class StridedSliceVertexTest {

    @Test
    public void doesOperateOnMatrix() {

        Slicer slicer = Slicer.builder()
            .all()
            .slice(1)
            .build();

        UnaryOperationTestHelpers.operatesOnInput(tensor -> tensor.slice(slicer), vertex -> vertex.slice(slicer));
    }

    @Test
    public void changesMatchGradientWithMatrixForward() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 3}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.slice(":,2");

        finiteDifferenceMatchesForwardModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-6);
    }

    @Test
    public void changesMatchGradientWithMatrixReverse() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 3}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.slice(":,2");

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-6);
    }

    @Test
    public void changesMatchGradientWithMatrixAndEllipsisForward() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 3, 4}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.slice("1,...,1:4:2");

        finiteDifferenceMatchesForwardModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-6);
    }

    @Test
    public void changesMatchGradientWithMatrixAndEllipsisReverse() {
        UniformVertex inputVertex = new UniformVertex(new long[]{2, 3, 4}, -10.0, 10.0);
        DoubleVertex outputVertex = inputVertex.slice("1,...,1:4:2");

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-6);
    }

    @Test
    public void changesMatchGradientWithCompletelyInsulatedSliceForward() {
        UniformVertex inputVertex = new UniformVertex(new long[]{3, 4}, -10.0, 10.0);

        DoubleVertex outputVertex = inputVertex
            .times(ConstantVertex.of(DoubleTensor.arange(2).reshape(2, 1, 1)))
            .slice(":,0:3:2,3")
            .sum(1);

        finiteDifferenceMatchesForwardModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-6);
    }

    @Test
    public void changesMatchGradientWithCompletelyInsulatedSliceReverse() {
        UniformVertex inputVertex = new UniformVertex(new long[]{3, 4}, -10.0, 10.0);

        DoubleVertex outputVertex = inputVertex
            .times(ConstantVertex.of(DoubleTensor.arange(2).reshape(2, 1, 1)))
            .slice(":,0:3:2,3")
            .sum(1);

        finiteDifferenceMatchesReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-6);
    }

    @Test
    public void changesMatchGradientWithCompletelyInsulatedSlice() {
        UniformVertex inputVertex = new UniformVertex(new long[]{3, 4, 5}, -10.0, 10.0);

        DoubleVertex outputVertex = inputVertex
            .times(ConstantVertex.of(DoubleTensor.arange(6).reshape(2, 3, 1, 1)))
            .slice("1,...,0:3:2,3:4")
            .sum(2);

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputVertex), outputVertex, 0.001, 1e-6);
    }
}
