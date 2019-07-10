package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;
import org.junit.Assert;
import org.junit.Test;

public class IntegerReshapeVertexTest {

    @Test
    public void reshapeVertex() {
        IntegerVertex a = new PoissonVertex(new long[]{2, 2}, 0.5);
        a.setValue(IntegerTensor.create(new int[]{1, 2, 3, 4}, 2, 2));

        IntegerVertex reshapeVertex = a.reshape(4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new long[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(new int[]{1, 2, 3, 4}, reshapeVertex.getValue().asFlatIntegerArray());
    }
}