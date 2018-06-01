package io.improbable.keanu.vertices.generic;

import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPT;
import io.improbable.keanu.vertices.generic.nonprobabilistic.CPTVertex;
import org.junit.Test;

import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex.FALSE;
import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex.TRUE;
import static junit.framework.TestCase.assertEquals;

public class CPTVertexTest {

    @Test
    public void canSetAndCascade() {

        BoolVertex A = new Flip(0.2);

        CPTVertex<Boolean> cpt = CPT.of(A)
            .when(true).then(TRUE)
            .orDefault(FALSE);

        A.setAndCascade(false);
        assertEquals(false, cpt.getValue().booleanValue());

        A.setAndCascade(true);
        assertEquals(true, cpt.getValue().booleanValue());
    }

    @Test
    public void canBeXOR() {

        BoolVertex A = new Flip(0.5);
        BoolVertex B = new Flip(0.5);

        CPTVertex<Boolean> cpt = CPT.of(A, B)
            .when(true, true).then(FALSE)
            .when(false, true).then(TRUE)
            .when(true, false).then(TRUE)
            .orDefault(FALSE);

        A.setAndCascade(true);
        B.setAndCascade(true);
        assertEquals(false, cpt.getValue().booleanValue());

        A.setAndCascade(false);
        B.setAndCascade(true);
        assertEquals(true, cpt.getValue().booleanValue());

        A.setAndCascade(true);
        B.setAndCascade(false);
        assertEquals(true, cpt.getValue().booleanValue());

        A.setAndCascade(false);
        B.setAndCascade(false);
        assertEquals(false, cpt.getValue().booleanValue());
    }
}
