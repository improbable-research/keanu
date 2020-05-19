package io.improbable.keanu.vertices.tensor.number.fixed.intgr.nonprobabilistic.operators.unary;

import io.improbable.keanu.tensor.intgr.IntegerTensor;
import io.improbable.keanu.vertices.tensor.number.fixed.intgr.IntegerVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.DoubleVertex;
import io.improbable.keanu.vertices.tensor.number.floating.dbl.nonprobabilistic.ConstantDoubleVertex;
import org.junit.Test;

import static io.improbable.keanu.tensor.TensorMatchers.valuesAndShapesMatch;
import static org.hamcrest.MatcherAssert.assertThat;

public class NaNArgMinVertexTest {

    @Test
    public void findsNaNArgMin() {
        DoubleVertex a = new ConstantDoubleVertex(1, 5, Double.NaN, 4).reshape(2, 2);

        IntegerVertex totalArgMin = a.nanArgMin();
        IntegerVertex dim1ArgMin = a.nanArgMin(1);

        assertThat(IntegerTensor.scalar(0), valuesAndShapesMatch(totalArgMin.getValue()));
        assertThat(IntegerTensor.create(0, 1), valuesAndShapesMatch(dim1ArgMin.getValue()));
    }
}
