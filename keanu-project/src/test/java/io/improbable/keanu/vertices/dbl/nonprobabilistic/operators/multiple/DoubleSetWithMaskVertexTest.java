package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import static org.junit.Assert.assertArrayEquals;
import org.junit.Before;
import org.junit.Test;

public class DoubleSetWithMaskVertexTest {

    DoubleVertex vertexA;

    @Before
    public void setup() {
        vertexA = ConstantVertex.of(new double[] {1., 2., 3., 4.}, 2, 2);
    }

    @Test
    public void canSetWithMaskGivenScalar() {
        DoubleVertex mask = vertexA.toGreaterThanMask(ConstantVertex.of(new double[]{2., 2., 2., 2.}, 2, 2));
        DoubleVertex result = new DoubleSetWithMaskVertex(vertexA, mask, ConstantVertex.of(-2.));

        assertArrayEquals(new double[]{1., 2., -2, -2}, result.getValue().asFlatDoubleArray(), 0.0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void cannotSetWithMaskGivenMatrix() {
        DoubleVertex mask = vertexA.toGreaterThanMask(ConstantVertex.of(new double[]{2., 2., 2., 2.}, 2, 2));
        DoubleVertex result = new DoubleSetWithMaskVertex(vertexA, mask, ConstantVertex.of(4., -2.));
    }

    /**
     * Zero is a special case because it's usually the value that the mask uses to mean "false"
     */
    @Test
    public void canSetToZero() {
        DoubleVertex mask = vertexA.toLessThanMask(ConstantVertex.of(new double[]{2., 2., 2., 2.}, 2, 2));
        DoubleVertex result = new DoubleSetWithMaskVertex(vertexA, mask, ConstantVertex.of(0.));

        assertArrayEquals(new double[]{0., 2., 3., 4.}, result.getValue().asFlatDoubleArray(), 0.0);
    }
}