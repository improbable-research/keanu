package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.unary;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.Differentiator;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;

public class SumVertexTest {

    @Test
    public void doesSum() {

        DoubleVertex in = new DistributionVertexBuilder()
            .shaped(1, 5)
            .withInput(ParameterName.MIN, 0.)
            .withInput(ParameterName.MAX, 10.)
            .uniform();
        in.setValue(new double[]{1, 2, 3, 4, 5});

        DoubleVertex summed = in.sum();

        DoubleTensor wrtIn = new Differentiator().calculateDual(summed).getPartialDerivatives().withRespectTo(in);
        DoubleTensor expectedWrtIn = DoubleTensor.ones(1, 1, 1, 5);

        assertEquals(1 + 2 + 3 + 4 + 5, summed.lazyEval().scalar(), 1e-5);
        assertArrayEquals(
            expectedWrtIn.asFlatDoubleArray(),
            wrtIn.asFlatDoubleArray(),
            1e-5
        );
        assertArrayEquals(expectedWrtIn.getShape(), wrtIn.getShape());
    }
}
