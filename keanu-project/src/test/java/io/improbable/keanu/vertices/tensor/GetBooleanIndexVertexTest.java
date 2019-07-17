package io.improbable.keanu.vertices.tensor;

import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class GetBooleanIndexVertexTest {

    @Test
    public void canGetBooleans() {
        DoubleVertex a = new ConstantDoubleVertex(-1, -2, 0, 3).reshape(2, 2);
        DoubleVertex result = a.get(a.greaterThan(0.0));
        assertThat(DoubleTensor.create(3), valuesAndShapesMatch(result.getValue()));
    }
}
