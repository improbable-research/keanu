package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.intgr.IntegerVertex;

public class IntegerSumVertexTest {

    @Test
    public void doesSum() {

        IntegerVertex in = new DistributionVertexBuilder()
            .shaped(1, 5)
            .withInput(ParameterName.MIN, 0)
            .withInput(ParameterName.MAX, 10)
            .uniformInt();
        in.setValue(new int[]{1, 2, 3, 4, 5});
        IntegerVertex summed = in.sum();

        assertEquals(1 + 2 + 3 + 4 + 5, summed.eval().scalar().intValue());
    }
}
