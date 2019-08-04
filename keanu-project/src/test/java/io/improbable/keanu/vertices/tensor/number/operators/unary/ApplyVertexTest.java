package io.improbable.keanu.vertices.tensor.number.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class ApplyVertexTest {

    @Test
    public void canApply() {
        DoubleVertex a = new ConstantDoubleVertex(1, -2, 7, 4);
        DoubleVertex result = a.apply(v -> v - 1);
        assertThat(DoubleTensor.scalar(-3), valuesAndShapesMatch(result.min().getValue()));
    }
}
