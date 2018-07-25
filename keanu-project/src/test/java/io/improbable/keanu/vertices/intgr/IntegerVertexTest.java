package io.improbable.keanu.vertices.intgr;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.function.Function;

import org.junit.Before;
import org.junit.Test;

import io.improbable.keanu.distributions.dual.ParameterName;
import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.DistributionVertexBuilder;
import io.improbable.keanu.vertices.dbl.probabilistic.VertexOfType;
import io.improbable.keanu.vertices.intgr.probabilistic.PoissonVertex;

public class IntegerVertexTest {

    IntegerVertex v1;
    IntegerVertex v2;

    @Before
    public void setup() {
        v1 = ConstantVertex.of(3);
        v2 = ConstantVertex.of(2);
    }

    @Test
    public void doesMultiply() {
        IntegerVertex result = v1.multiply(v2);
        result.eval();
        Integer expected = 6;
        assertEquals(result.getValue().scalar(), expected);
    }

    @Test
    public void doesAdd() {
        IntegerVertex result = v1.plus(v2);
        result.eval();
        Integer expected = 5;
        assertEquals(result.getValue().scalar(), expected);
    }

    @Test
    public void doesSubtract() {
        IntegerVertex result = v1.minus(v2);
        result.eval();
        Integer expected = 1;
        assertEquals(result.getValue().scalar(), expected);
    }

    @Test
    public void doesObserve() {
        PoissonVertex testIntegerVertex = VertexOfType.poisson(1.0);
        testIntegerVertex.observe(IntegerTensor.scalar(5));

        Integer expected = 5;
        assertEquals(testIntegerVertex.getValue().scalar(), expected);
        assertTrue(testIntegerVertex.isObserved());
    }

    @Test
    public void doesLambda() {
        Function<IntegerTensor, IntegerTensor> op = val -> val.plus(5);

        IntegerVertex result = v1.lambda(op);
        result.eval();
        Integer expected = 8;
        assertEquals(result.getValue().scalar(), expected);
    }

    @Test
    public void canObserveArrayOfValues() {
        IntegerVertex binomialVertex = VertexOfType.binomial(0.5, 20);
        int[] observation = new int[]{1, 2, 3};
        binomialVertex.observe(observation);
        assertArrayEquals(observation, binomialVertex.getValue().asFlatIntegerArray());
    }

    @Test
    public void canSetAndCascadeArrayOfValues() {
        IntegerVertex binomialVertex = VertexOfType.binomial(0.5, 20);
        int[] values = new int[]{1, 2, 3};
        binomialVertex.setAndCascade(values);
        assertArrayEquals(values, binomialVertex.getValue().asFlatIntegerArray());
    }

    @Test
    public void canSetValueAsArrayOfValues() {
        IntegerVertex binomialVertex = VertexOfType.binomial(0.5, 20);
        int[] values = new int[]{1, 2, 3};
        binomialVertex.setValue(values);
        assertArrayEquals(values, binomialVertex.getValue().asFlatIntegerArray());
    }

    @Test
    public void canSetValueAsScalarOnNonScalarVertex() {
        IntegerVertex binomialVertex = new DistributionVertexBuilder()
            .shaped(2, 1)
            .withInput(ParameterName.P, 0.5)
            .withInput(ParameterName.N, 20)
            .binomial();
        binomialVertex.setValue(2);
        assertArrayEquals(new int[]{2}, binomialVertex.getValue().asFlatIntegerArray());
    }

    @Test
    public void canSetAndCascadeAsScalarOnNonScalarVertex() {
        IntegerVertex binomialVertex = new DistributionVertexBuilder()
            .shaped(2, 1)
            .withInput(ParameterName.P, 0.5)
            .withInput(ParameterName.N, 20)
            .binomial();
        binomialVertex.setAndCascade(2);
        assertArrayEquals(new int[]{2}, binomialVertex.getValue().asFlatIntegerArray());
    }

}
