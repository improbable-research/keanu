package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.bool.BooleanVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class NumericalEqualsVertexTest {

    @Test
    public void canCompareNumbersNotEqual() {

        BooleanVertex equals = new NumericalEqualsVertex<>(
            ConstantVertex.of(1., 2., 3.),
            ConstantVertex.of(4., 5., 6.),
            0.0
        );

        BooleanTensor equality = equals.lazyEval();

        assertArrayEquals(new Boolean[]{false, false, false}, equality.asFlatArray());
    }

    @Test
    public void canCompareNumbersEqual() {

        BooleanVertex equals = new NumericalEqualsVertex<>(
            ConstantVertex.of(1., 2., 3.),
            ConstantVertex.of(1., 2., 3.),
            0.0
        );

        BooleanTensor equality = equals.lazyEval();

        assertArrayEquals(new Boolean[]{true, true, true}, equality.asFlatArray());
    }

    @Test
    public void canCompareNumbersAlmostEqual() {

        BooleanVertex equals = new NumericalEqualsVertex<>(
            ConstantVertex.of(1., 2., 3.),
            ConstantVertex.of(1.5, 2.01, 3),
            0.1
        );

        BooleanTensor equality = equals.lazyEval();

        assertArrayEquals(new Boolean[]{false, true, true}, equality.asFlatArray());
    }

    @Test
    public void canCompareIntegersAlmostEqual() {

        BooleanVertex equals = new NumericalEqualsVertex<>(
            ConstantVertex.of(1, 2, 3),
            ConstantVertex.of(2, 2, 6),
            1
        );

        BooleanTensor equality = equals.lazyEval();

        assertArrayEquals(new Boolean[]{true, true, false}, equality.asFlatArray());
    }

    @Test
    public void canCompareIntegersAndDoublesAlmostEqual() {

        BooleanVertex equals = new NumericalEqualsVertex<>(
            ConstantVertex.of(new int[]{1, 2, 3}).toDouble(),
            ConstantVertex.of(2.0, 2.0, 5.5),
            2.1
        );

        BooleanTensor equality = equals.lazyEval();

        assertArrayEquals(new Boolean[]{true, true, false}, equality.asFlatArray());
    }
}
