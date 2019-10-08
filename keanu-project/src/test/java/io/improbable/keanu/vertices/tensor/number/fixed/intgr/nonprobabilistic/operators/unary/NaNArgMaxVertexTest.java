package io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class NaNArgMaxVertexTest {

    @Test
    public void findsNaNArgMax() {
        DoubleVertex a = new ConstantDoubleVertex(1, 5, Double.NaN, 4).reshape(2, 2);

        IntegerVertex totalArgMax = a.nanArgMax();
        IntegerVertex dim1ArgMax = a.nanArgMax(1);

        assertThat(IntegerTensor.scalar(1), valuesAndShapesMatch(totalArgMax.getValue()));
        assertThat(IntegerTensor.create(1, 1), valuesAndShapesMatch(dim1ArgMax.getValue()));
    }
}
