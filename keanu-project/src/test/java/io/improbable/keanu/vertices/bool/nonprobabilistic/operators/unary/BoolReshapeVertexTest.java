package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import org.junit.Assert;
import org.junit.Test;

public class BoolReshapeVertexTest {

    @Test
    public void reshapeVertexWorksAsExpected() {
        BoolVertex a = new Flip(new int[]{2, 2}, 0.5);
        double[] aValues = a.getValue().asFlatDoubleArray();

        BoolReshapeVertex reshapeVertex = new BoolReshapeVertex(a, 4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new int[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(aValues, reshapeVertex.getValue().asFlatDoubleArray(), 1e-6);
    }

}
