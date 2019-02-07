package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Test;

import java.util.Map;

import static junit.framework.TestCase.assertEquals;

public class TreeTest {

    @Test
    public void canLogSumExp() {

        double a = -0.25;
        double b = 0.5;
        double expected = Math.log(Math.exp(a) + Math.exp(b));
        double actual = Tree.logSumExp(a, b);

        assertEquals(expected, actual, 1e-6);
    }

    @Test
    public void canAddTwoMapsOfValues() {

        DoubleVertex A = new GaussianVertex(new long[]{2}, 0, 1);
        DoubleVertex B = new GaussianVertex(new long[]{2}, 0, 1);

        Map<VariableReference, DoubleTensor> left = ImmutableMap.of(
            A.getReference(), DoubleTensor.create(1, 2),
            B.getReference(), DoubleTensor.create(3, 4)
        );

        Map<VariableReference, DoubleTensor> right = ImmutableMap.of(
            A.getReference(), DoubleTensor.create(5, 6),
            B.getReference(), DoubleTensor.create(7, 8)
        );

        Map<VariableReference, DoubleTensor> sum = Tree.add(left, right);

        DoubleTensor expectedA = DoubleTensor.create(6, 8);
        DoubleTensor expectedB = DoubleTensor.create(10, 12);

        assertEquals(expectedA, sum.get(A.getReference()));
        assertEquals(expectedB, sum.get(B.getReference()));
    }

}
