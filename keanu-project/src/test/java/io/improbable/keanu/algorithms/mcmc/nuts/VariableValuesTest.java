package io.improbable.keanu.algorithms.mcmc.nuts;

import com.google.common.collect.ImmutableMap;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.GaussianVertex;
import org.junit.Before;
import org.junit.Test;

import java.util.Map;

import static junit.framework.TestCase.assertEquals;

public class VariableValuesTest {

    private VariableReference AReference;
    private VariableReference BReference;

    private Map<VariableReference, DoubleTensor> left;
    private Map<VariableReference, DoubleTensor> right;

    @Before
    public void setup() {
        DoubleVertex A = new GaussianVertex(new long[]{2}, 0, 1);
        AReference = A.getReference();

        DoubleVertex B = new GaussianVertex(new long[]{2}, 0, 1);
        BReference = B.getReference();

        left = ImmutableMap.of(
            A.getReference(), DoubleTensor.create(1, 2),
            B.getReference(), DoubleTensor.create(3, 4)
        );

        right = ImmutableMap.of(
            A.getReference(), DoubleTensor.create(5, 6),
            B.getReference(), DoubleTensor.create(7, 8)
        );
    }

    @Test
    public void canAddTwoMapsOfValues() {

        Map<VariableReference, DoubleTensor> result = VariableValues.add(left, right);

        DoubleTensor expectedA = DoubleTensor.create(6, 8);
        DoubleTensor expectedB = DoubleTensor.create(10, 12);

        assertEquals(expectedA, result.get(AReference));
        assertEquals(expectedB, result.get(BReference));
    }

    @Test
    public void canTimesTwoMapsOfValues() {

        Map<VariableReference, DoubleTensor> result = VariableValues.times(left, right);

        DoubleTensor expectedA = DoubleTensor.create(5, 12);
        DoubleTensor expectedB = DoubleTensor.create(21, 32);

        assertEquals(expectedA, result.get(AReference));
        assertEquals(expectedB, result.get(BReference));
    }

    @Test
    public void canTimesAMapOfValues() {

        Map<VariableReference, DoubleTensor> result = VariableValues.times(left, 2);

        DoubleTensor expectedA = DoubleTensor.create(2, 4);
        DoubleTensor expectedB = DoubleTensor.create(6, 8);

        assertEquals(expectedA, result.get(AReference));
        assertEquals(expectedB, result.get(BReference));
    }

    @Test
    public void canDivideAMapOfValues() {

        Map<VariableReference, DoubleTensor> result = VariableValues.divide(left, 2.0);

        DoubleTensor expectedA = DoubleTensor.create(0.5, 1.0);
        DoubleTensor expectedB = DoubleTensor.create(1.5, 2.0);

        assertEquals(expectedA, result.get(AReference));
        assertEquals(expectedB, result.get(BReference));
    }

    @Test
    public void canPowAMapOfValues() {

        Map<VariableReference, DoubleTensor> result = VariableValues.pow(left, 2);

        DoubleTensor expectedA = DoubleTensor.create(1, 4);
        DoubleTensor expectedB = DoubleTensor.create(9, 16);

        assertEquals(expectedA, result.get(AReference));
        assertEquals(expectedB, result.get(BReference));
    }

    @Test
    public void canDotProductTwoMapsOfValues() {

        double result = VariableValues.dotProduct(left, right);

        double expected = 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8;

        assertEquals(expected, result, 1e-6);
    }

    @Test
    public void canZeros() {

        Map<VariableReference, DoubleTensor> result = VariableValues.zeros(left);

        DoubleTensor expectedA = DoubleTensor.create(0, 0);
        DoubleTensor expectedB = DoubleTensor.create(0, 0);

        assertEquals(expectedA, result.get(AReference));
        assertEquals(expectedB, result.get(BReference));
    }

    @Test
    public void canOnes() {

        Map<VariableReference, DoubleTensor> result = VariableValues.ones(left);

        DoubleTensor expectedA = DoubleTensor.create(1, 1);
        DoubleTensor expectedB = DoubleTensor.create(1, 1);

        assertEquals(expectedA, result.get(AReference));
        assertEquals(expectedB, result.get(BReference));
    }

}
