package io.improbable.keanu.vertices.generictensor.nonprobabilistic;

import io.improbable.keanu.tensor.bool.BooleanTensor;
import io.improbable.keanu.vertices.booltensor.BoolVertex;
import io.improbable.keanu.vertices.booltensor.probabilistic.Flip;
import org.junit.Test;

import static io.improbable.keanu.vertices.booltensor.BoolVertex.FALSE;
import static io.improbable.keanu.vertices.booltensor.BoolVertex.TRUE;
import static junit.framework.TestCase.assertFalse;
import static junit.framework.TestCase.assertTrue;

public class CPTVertexTest {

    @Test
    public void canSetAndCascade() {

        BoolVertex A = new Flip(0.2);

        CPTVertex<Boolean, BooleanTensor> cpt = CPT.of(A)
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

        CPTVertex<Boolean, BooleanTensor> cpt = CPT.of(A, B)
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
