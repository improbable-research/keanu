package io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.ConstantIntegerVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class ArgMaxVertexTest {

    @Test
    public void findsArgMax() {
        IntegerVertex a = new ConstantIntegerVertex(1, 5, 3, 4).reshape(2, 2);

        IntegerVertex totalArgMax = a.argMax();
        IntegerVertex dim1ArgMax = a.argMax(1);

        assertThat(IntegerTensor.scalar(1), valuesAndShapesMatch(totalArgMax.getValue()));
        assertThat(IntegerTensor.create(1, 1), valuesAndShapesMatch(dim1ArgMax.getValue()));
    }
}
