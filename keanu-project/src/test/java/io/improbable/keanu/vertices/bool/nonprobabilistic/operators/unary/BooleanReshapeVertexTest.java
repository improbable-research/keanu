package io.improbable.keanu.vertices.bool.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import io.improbable.keanu.vertices.bool.probabilistic.BernoulliVertex;
import org.junit.Assert;
import org.junit.Test;

public class BooleanReshapeVertexTest {

    @Test
    public void reshapeVertexWorksAsExpected() {
        BooleanVertex a = new BernoulliVertex(0.5);
        a.setValue(BooleanTensor.create(new boolean[]{true, true, false, false}, 2, 2));

        BooleanReshapeVertex reshapeVertex = new BooleanReshapeVertex(a, 4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new long[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(new int[]{1, 1, 0, 0}, reshapeVertex.getValue().asFlatIntegerArray());
    }

}
