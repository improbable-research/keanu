package io.improbable.keanu.vertices;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import org.junit.Test;

public class AssertVertexTest {

    @Test
    public void basicAssertVertexTest() {
        ConstantBoolVertex constBool = new ConstantBoolVertex(BooleanTensor.create(true));
        AssertVertex assertVertex = new AssertVertex(constBool, BooleanTensor.create(false));
        assertVertex.calculate();
    }
}
