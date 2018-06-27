package io.improbable.keanu.vertices.bool.nonprobabilistic.operators;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;

public class NumericalEqualsVertexTest {

    @Test
    public void canNumbersNotEqual() {

        NumericalEqualsVertex equals = new NumericalEqualsVertex(
            ConstantVertex.of(new double[]{1, 2, 3}),
            ConstantVertex.of(new double[]{4, 5, 6}),
            ConstantVertex.of(new double[]{0, 0, 0})
        );

        BooleanTensor equality = equals.lazyEval();

        assertArrayEquals(new Boolean[]{false, false, false}, equality.asFlatArray());
    }
}
