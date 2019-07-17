package io.improbable.keanu.vertices.tensor.number.floating.operators.unary;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class ReplaceNaNVertexTest {

    @Test
    public void doesOperateOnMatrix() {
        DoubleVertex a = new ConstantDoubleVertex(1.0, -2, 0, Double.NaN, -0);
        DoubleVertex result = a.replaceNaN(0.0);

        assertThat(DoubleTensor.create(1.0, -2.0, 0, 0.0, -0), valuesAndShapesMatch(result.getValue()));
    }
}
