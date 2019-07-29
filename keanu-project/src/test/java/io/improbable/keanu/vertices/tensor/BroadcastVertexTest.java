package io.improbable.keanu.vertices.tensor;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

public class BroadcastVertexTest {

    @Test
    public void changesMatchGradient() {
        UniformVertex inputA = new UniformVertex(new long[]{2, 5, 1}, -10.0, 10.0);
        DoubleVertex broadcast = inputA.broadcast(3, 2, 5, 2);
        DoubleVertex outputVertex = broadcast.times(
            new ConstantDoubleVertex(DoubleTensor.arange(2))
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }
}
