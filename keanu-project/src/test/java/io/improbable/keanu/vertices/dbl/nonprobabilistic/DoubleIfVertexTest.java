package io.improbable.keanu.vertices.dbl.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import org.junit.Test;

public class DoubleIfVertexTest {

    @Test
    public void caneDoIf() {
        BoolVertex bool = new ConstantBoolVertex(BooleanTensor.create(new boolean[]{true, true}, 1, 2));

        If.isTrue(bool)
            .then(1)
            .orElse(2);
    }

}
