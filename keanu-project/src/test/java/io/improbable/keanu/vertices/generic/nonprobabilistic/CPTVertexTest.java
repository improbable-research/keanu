package io.improbable.keanu.vertices.generic.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.bool.BoolVertex;
import io.improbable.keanu.vertices.bool.probabilistic.Flip;
import org.junit.Test;

import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex.FALSE;
import static io.improbable.keanu.vertices.bool.nonprobabilistic.ConstantBoolVertex.TRUE;
import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;

public class CPTVertexTest {

    @Test
    public void canSetAndCascade() {

        BoolVertex A = new Flip(0.2);

        CPTVertex<BooleanTensor> cpt = CPT.of(A)
            .when(true).then(TRUE)
            .orDefault(FALSE);

        A.setAndCascade(false);
        assertFalse(cpt.getValue().scalar());

        A.setAndCascade(true);
        assertTrue(cpt.getValue().scalar());
    }

    @Test
    public void canBeXOR() {

        BoolVertex A = new Flip(0.5);
        BoolVertex B = new Flip(0.5);

        CPTVertex<BooleanTensor> cpt = CPT.of(A, B)
            .when(true, true).then(FALSE)
            .when(false, true).then(TRUE)
            .when(true, false).then(TRUE)
            .orDefault(FALSE);

        A.setAndCascade(true);
        B.setAndCascade(true);
        assertFalse(cpt.getValue().scalar());

        A.setAndCascade(false);
        B.setAndCascade(true);
        assertTrue(cpt.getValue().scalar());

        A.setAndCascade(true);
        B.setAndCascade(false);
        assertTrue(cpt.getValue().scalar());

        A.setAndCascade(false);
        B.setAndCascade(false);
        assertFalse(cpt.getValue().scalar());
    }
}
