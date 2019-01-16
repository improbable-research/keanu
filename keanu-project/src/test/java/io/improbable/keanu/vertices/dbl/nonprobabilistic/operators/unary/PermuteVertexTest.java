package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.UniformVertex;

public class PermuteVertexTest {

    @Test
    public void canPermuteForTranpose() {
        DoubleVertex a = new UniformVertex(0, 10);
        a.setValue(DoubleTensor.create(new double[]{1, 2, 3, 4}, 2, 2));

        PermuteVertex PermuteVertex = new PermuteVertex(a, 1, 0);
        PermuteVertex.getValue();

        Assert.assertArrayEquals(new long[]{2, 2}, PermuteVertex.getShape());
        Assert.assertArrayEquals(a.getValue().transpose().asFlatDoubleArray(), PermuteVertex.getValue().asFlatDoubleArray(), 1e-6);
    }


}
