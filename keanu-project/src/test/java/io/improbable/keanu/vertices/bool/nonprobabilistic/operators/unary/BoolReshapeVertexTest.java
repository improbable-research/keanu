package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import org.junit.Assert;
import org.junit.Test;

public class BoolReshapeVertexTest {

    @Test
    public void reshapeVertexWorksAsExpected() {
        BoolVertex a = new Flip(0.5);
        a.setValue(BooleanTensor.create(new boolean[]{true, true, false, false}, 2, 2));

        BoolReshapeVertex reshapeVertex = new BoolReshapeVertex(a, 4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new int[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(new int[]{1, 1, 0, 0}, reshapeVertex.getValue().asFlatIntegerArray());
    }

}
