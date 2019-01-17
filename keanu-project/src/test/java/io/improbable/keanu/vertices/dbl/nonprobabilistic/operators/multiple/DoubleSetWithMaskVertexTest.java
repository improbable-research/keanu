package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import static org.junit.Assert.assertArrayEquals;
import org.junit.Before;
import org.junit.Test;

public class DoubleSetWithMaskVertexTest {

    private DoubleVertex vertex;

    @Before
    public void setup() {
        vertex = ConstantVertex.of(new double[] {1., 2., 3., 4.}, 2, 2);
    }

    @Test(expected = IllegalArgumentException.class)
    public void operandAndMaskMustBeSameShape() {
        DoubleVertex mask = ConstantVertex.of(new double[] {1., 2., 3., 4.}, 4, 1);
        DoubleVertex result = new DoubleSetWithMaskVertex(vertex, mask, ConstantVertex.of(-.2));
    }

    @Test
    public void canSetWithMaskGivenScalar() {
        DoubleVertex mask = vertex.toGreaterThanMask(ConstantVertex.of(new double[]{2., 2., 2., 2.}, 2, 2));
        DoubleVertex result = new DoubleSetWithMaskVertex(vertex, mask, ConstantVertex.of(-2.));

        assertArrayEquals(new double[]{1., 2., -2, -2}, result.getValue().asFlatDoubleArray(), 0.0);
    }

    @Test
    public void canSetWithMaskGivenMatrixButOnlyTakesItsScalarValue() {
        DoubleVertex mask = vertex.toGreaterThanMask(ConstantVertex.of(new double[]{2., 2., 2., 2.}, 2, 2));
        DoubleVertex result = new DoubleSetWithMaskVertex(vertex, mask, ConstantVertex.of(4., -2.));

        assertArrayEquals(new double[]{1., 2., 4., 4.}, result.getValue().asFlatDoubleArray(), 0.0);
    }

    /**
     * Zero is a special case because it's usually the value that the mask uses to mean "false"
     */
    @Test
    public void canSetToZero() {
        DoubleVertex mask = vertex.toLessThanMask(ConstantVertex.of(new double[]{2., 2., 2., 2.}, 2, 2));
        DoubleVertex result = new DoubleSetWithMaskVertex(vertex, mask, ConstantVertex.of(0.));

        assertArrayEquals(new double[]{0., 2., 3., 4.}, result.getValue().asFlatDoubleArray(), 0.0);
    }
}