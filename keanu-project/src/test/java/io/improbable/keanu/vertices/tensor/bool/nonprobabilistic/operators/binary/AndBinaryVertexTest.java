package io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.operators.binary;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.tensor.bool.BooleanVertex;
import io.improbable.keanu.vertices.tensor.bool.nonprobabilistic.ConstantBooleanVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class AndBinaryVertexTest {

    @Test
    public void canAndTwoBooleans() {
        BooleanVertex a = new ConstantBooleanVertex(true, false, true, true).reshape(2, 2);
        BooleanVertex b = new ConstantBooleanVertex(true, false, false, true).reshape(2, 2);
        BooleanVertex result = a.and(b);

        assertThat(BooleanTensor.create(true, false, false, true).reshape(2, 2), valuesAndShapesMatch(result.getValue()));
    }
}
