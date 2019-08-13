package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.ConstantVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class TriUpperVertexTest {

    @Test
    public void doesSimpleTriUpper() {
        DoubleVertex input = ConstantVertex.of(1., 2., 3., 4.).reshape(2, 2);
        DoubleVertex result = input.triUpper(0);

        assertThat(result.getValue(), valuesAndShapesMatch(DoubleTensor.create(1, 2, 0, 4).reshape(2, 2)));
    }
}
