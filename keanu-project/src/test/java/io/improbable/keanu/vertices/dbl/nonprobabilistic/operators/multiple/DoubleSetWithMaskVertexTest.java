package io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.multiple;

import io.improbable.keanu.tensor.TensorMatchers;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

import static org.junit.Assert.assertThat;

public class DoubleSetWithMaskVertexTest {

    private DoubleVertex vertex;

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Before
    public void setup() {
        vertex = ConstantVertex.of(new double[] {1., 2., 3., 4.}, 2, 2);
    }

    @Test
    public void operandAndMaskMustBeSameShape() {
        DoubleVertex mask = ConstantVertex.of(1.);
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("Shapes must match");
        DoubleVertex result = new DoubleSetWithMaskVertex(vertex, mask, ConstantVertex.of(-2.));
    }

    @Test
    public void setValueMustBeScalar() {
        DoubleVertex mask = ConstantVertex.of(new double[] {1., 2., 3., 4.}, 2, 2);
        thrown.expect(IllegalArgumentException.class);
        thrown.expectMessage("setValue must be scalar");
        DoubleVertex result = new DoubleSetWithMaskVertex(vertex, mask, ConstantVertex.of(-2., -2.));
    }

    @Test
    public void canSetWithMaskGivenScalar() {
        DoubleVertex mask = vertex.toGreaterThanMask(ConstantVertex.of(new double[]{2., 2., 2., 2.}, 2, 2));
        DoubleVertex result = new DoubleSetWithMaskVertex(vertex, mask, ConstantVertex.of(-2.));
        DoubleTensor expected = DoubleTensor.create(new double[] {1., 2., -2., -2.}, 2, 2);
        assertThat(expected, TensorMatchers.valuesAndShapesMatch(result.getValue()));
    }

    /**
     * Zero is a special case because it's usually the value that the mask uses to mean "false"
     */
    @Test
    public void canSetToZero() {
        DoubleVertex mask = vertex.toLessThanMask(ConstantVertex.of(new double[]{2., 2., 2., 2.}, 2, 2));
        DoubleVertex result = new DoubleSetWithMaskVertex(vertex, mask, ConstantVertex.of(0.));
        DoubleTensor expected = DoubleTensor.create(new double[] {0., 2., 3., 4.}, 2, 2);
        assertThat(expected, TensorMatchers.valuesAndShapesMatch(result.getValue()));
    }
}