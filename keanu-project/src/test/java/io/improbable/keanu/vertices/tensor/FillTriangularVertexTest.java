package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class FillTriangularVertexTest {

    @Test
    public void doesSimpleUpperAndLower() {
        DoubleVertex input = ConstantVertex.of(1., 2., 3.);
        DoubleVertex result = input.fillTriangular(true, true);

        assertThat(result.getValue(), valuesAndShapesMatch(DoubleTensor.create(1, 2, 2, 3).reshape(2, 2)));
    }
}
