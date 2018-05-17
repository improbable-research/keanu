package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.generic.nonprobabilistic.If;
import io.improbable.keanu.vertices.generic.nonprobabilistic.IfVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.bool.BoolVertex.FALSE;
import static io.improbable.keanu.vertices.bool.BoolVertex.TRUE;
import static junit.framework.TestCase.assertFalse;
import static org.junit.Assert.assertTrue;

public class IfVertexTest {

    @Test
    public void functionsAsIf() {

        BoolVertex predicate = new Flip(0.5);

        IfVertex<Boolean> ifIsTrue = If.isTrue(predicate)
            .then(TRUE)
            .orElse(FALSE);

        predicate.setAndCascade(true);
        assertTrue(ifIsTrue.getValue());

        predicate.setAndCascade(false);
        assertFalse(ifIsTrue.getValue());
    }
}
