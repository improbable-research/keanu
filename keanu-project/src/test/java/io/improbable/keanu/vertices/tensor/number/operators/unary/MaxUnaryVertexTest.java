package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class MaxUnaryVertexTest {

    @Test
    public void canGetMax() {
        DoubleVertex a = new ConstantDoubleVertex(1, 2, 7, 4);
        assertThat(DoubleTensor.scalar(7), valuesAndShapesMatch(a.max().getValue()));
    }
}
