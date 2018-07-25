package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;

public class BoolReshapeVertexTest {

    @Test
    public void reshapeVertexWorksAsExpected() {
        BooleanVertex a = new Flip(0.5);
        a.setValue(BooleanTensor.create(new boolean[]{true, true, false, false}, 2, 2));

        BooleanVertex reshapeVertex = a.reshape(4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new int[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(new int[]{1, 1, 0, 0}, reshapeVertex.getValue().asFlatIntegerArray());
    }

}
