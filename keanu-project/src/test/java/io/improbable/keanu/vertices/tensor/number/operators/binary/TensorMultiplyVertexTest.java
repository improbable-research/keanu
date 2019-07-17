package io.improbable.keanu.vertices.tensor.number.operators.binary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Ignore;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

public class TensorMultiplyVertexTest {

    @Test
    @Ignore
    public void changesMatchGradient() {
        UniformVertex inputA = new UniformVertex(new long[]{2, 5, 3}, -10.0, 10.0);
        UniformVertex inputB = new UniformVertex(new long[]{5, 2, 2}, -10.0, 10.0);
        DoubleVertex mmultVertex = inputA.tensorMultiply(inputB, new int[]{1}, new int[]{0});
        DoubleVertex outputVertex = mmultVertex.times(
            new ConstantDoubleVertex(DoubleTensor.arange(12.).reshape(3, 2, 2))
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA, inputB), outputVertex, INCREMENT, DELTA);
    }
}
