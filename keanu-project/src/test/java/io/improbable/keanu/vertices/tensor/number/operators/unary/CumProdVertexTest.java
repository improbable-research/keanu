package io.improbable.keanu.vertices.tensor.number.operators.unary;

import com.google.common.collect.ImmutableList;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.probabilistic.UniformVertex;
import org.junit.Ignore;
import org.junit.Test;

import static io.improbable.keanu.vertices.tensor.number.TensorTestOperations.finiteDifferenceMatchesForwardAndReverseModeGradient;

public class CumProdVertexTest {

    @Test
    @Ignore
    public void changesMatchGradient() {
        UniformVertex inputA = new UniformVertex(new long[]{2, 3, 4}, -10.0, 10.0);
        DoubleVertex cumProd = inputA.cumProd(1);
        DoubleVertex outputVertex = cumProd.times(
            new ConstantDoubleVertex(DoubleTensor.arange(4))
        );

        final double INCREMENT = 10;
        final double DELTA = 1e-10;

        finiteDifferenceMatchesForwardAndReverseModeGradient(ImmutableList.of(inputA), outputVertex, INCREMENT, DELTA);
    }
}
