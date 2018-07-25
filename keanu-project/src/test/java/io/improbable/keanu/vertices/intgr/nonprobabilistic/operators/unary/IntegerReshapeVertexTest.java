package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import org.junit.Assert;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerReshapeVertexTest {

    @Test
    public void reshapeVertex() {
        IntegerVertex a = new DistributionVertexBuilder().shaped(2, 2).withInput(ParameterName.MU, 0.5).poisson();
        a.setValue(IntegerTensor.create(new int[]{1, 2, 3, 4}, 2, 2));

        IntegerVertex reshapeVertex = a.reshape(4, 1);
        reshapeVertex.getValue();

        Assert.assertArrayEquals(new int[]{4, 1}, reshapeVertex.getShape());
        Assert.assertArrayEquals(new int[]{1, 2, 3, 4}, reshapeVertex.getValue().asFlatIntegerArray());
    }
}