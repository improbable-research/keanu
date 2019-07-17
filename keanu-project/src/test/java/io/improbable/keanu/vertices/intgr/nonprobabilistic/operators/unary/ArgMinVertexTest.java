package io.improbable.keanu.vertices.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.intgr.IntegerVertex;
import io.improbable.keanu.vertices.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class ArgMinVertexTest {

    @Test
    public void findsArgMin() {
        IntegerVertex a = new ConstantIntegerVertex(1, 5, 3, 4).reshape(2, 2);

        IntegerVertex totalArgMin = a.argMin();
        IntegerVertex dim0ArgMin = a.argMin(0);

        assertThat(IntegerTensor.scalar(0), valuesAndShapesMatch(totalArgMin.getValue()));
        assertThat(IntegerTensor.create(0, 1), valuesAndShapesMatch(dim0ArgMin.getValue()));
    }
}
